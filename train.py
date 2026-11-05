"""Reproducible training/validation entrypoint. Does not evaluate test data."""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import PIL
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from src.data import LeafDataset, seed_worker
from src.metrics import confusion_metrics
from src.models import build_model, profile_model

ROOT = Path(__file__).resolve().parent


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def lr_at(epoch, config):
    warm = config["warmup_epochs"]
    if epoch < warm:
        return config["learning_rate"] * (epoch + 1) / max(warm, 1)
    phase = (epoch - warm) / max(config["epochs"] - warm - 1, 1)
    floor = config["minimum_lr_ratio"]
    return config["learning_rate"] * (floor + (1 - floor) * (1 + math.cos(math.pi * phase)) / 2)


def run_epoch(model, loader, criterion, device, config, optimizer=None, scaler=None, epoch=0):
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros(config["num_classes"], config["num_classes"], dtype=torch.int64, device=device)
    loss_sum = torch.zeros((), device=device)
    count = 0
    dtype = torch.bfloat16 if config["amp"] == "bf16" else torch.float16
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for step, (images, labels, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=device.type == "cuda" and config["amp"] != "off"):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}, step {step + 1}")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        n = labels.size(0)
        loss_sum += loss.detach() * n
        count += n
        indexes = labels * config["num_classes"] + logits.detach().argmax(1)
        confusion += torch.bincount(indexes, minlength=config["num_classes"] ** 2).reshape(confusion.shape)
        if (step + 1) % 50 == 0:
            print(f"epoch={epoch + 1} {'train' if training else 'val'} step={step + 1}/{len(loader)} loss={loss_sum.item() / count:.4f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    metrics = confusion_metrics(confusion)
    metrics.update(loss=(loss_sum / count).item(), seconds=seconds,
                   samples=count, images_per_second=count / seconds)
    return metrics


def atomic_save(obj, path):
    temporary = path.with_suffix(".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_pilot_224.json")
    parser.add_argument("--stop-after-epochs", type=int,
                        help="Stop after this many total completed epochs; keeps full configured LR horizon")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--deadline-unix", type=float,
                        help="Save and stop at an epoch boundary after this absolute Unix time")
    parser.add_argument("--allow-loader-change", action="store_true",
                        help="Record and allow only a documented data-loader memory configuration change on resume")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    manifests = ROOT / "data/manifests"
    audit = json.loads((manifests / "audit.json").read_text(encoding="utf-8"))
    review = manifests / "near_duplicate_review.json"
    if not review.exists():
        raise RuntimeError("Run scripts/freeze_splits.py before training; manifests must be frozen")
    reviewed = json.loads(review.read_text(encoding="utf-8"))
    if audit["cross_split_near_duplicate_candidates"] or reviewed["remaining_cross_split_candidates"]:
        raise RuntimeError("Unresolved cross-split near-duplicate candidates")
    classes = json.loads((manifests / "classes.json").read_text(encoding="utf-8"))
    norm = json.loads((manifests / "normalization.json").read_text(encoding="utf-8"))
    config["num_classes"] = len(classes)
    config["normalization"] = norm
    config["manifest_sha256"] = {s: fingerprint(manifests / f"{s}.csv") for s in ("train", "val", "test")}
    for split, sha in config["manifest_sha256"].items():
        if sha != audit["manifest_sha256"][split] or sha != reviewed["manifest_sha256"][split]:
            raise RuntimeError(f"Frozen {split} manifest changed after audit")
    torch.set_num_threads(config["cpu_threads"])
    seed_everything(config["seed"], config["deterministic"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this GPU experiment configuration")
    if config["amp"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requested but unsupported")
    run = ROOT / "runs" / config["run_name"]
    run.mkdir(parents=True, exist_ok=True)
    if (run / "config.json").exists() and args.resume is None:
        raise RuntimeError(f"Run already exists: {run}; use --resume or a different run_name")
    if args.resume:
        old_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if old_config != config:
            different = {key for key in old_config.keys() | config.keys() if old_config.get(key) != config.get(key)}
            loader_keys = {"workers", "cpu_threads", "loader_prefetch", "pin_memory"}
            if not args.allow_loader_change or not different <= loader_keys:
                raise RuntimeError(f"Resume config/manifests changed: {different}; refusing incompatible continuation")
            changes = run / "loader_changes"
            changes.mkdir(exist_ok=True)
            dump(changes / f"{time.time_ns()}.json", {"old_config": old_config, "new_config": config,
                 "changed_fields": sorted(different), "reason": "Documented loader-only execution amendment; see configs/local_protocol.json and the archived protocol."})
            print(f"Documented loader-only resume change: {sorted(different)}", flush=True)
    dump(run / "config.json", config)
    (run / "command.txt").write_text(subprocess.list2cmdline([sys.executable] + sys.argv), encoding="utf-8")
    source_hashes = {str(p.relative_to(ROOT)): fingerprint(p) for p in sorted(ROOT.rglob("*.py"))
                     if not any(part in ("data", "runs", ".git") for part in p.relative_to(ROOT).parts)}
    dump(run / ("resume_source_sha256.json" if args.resume else "source_sha256.json"), source_hashes)
    dump(run / "environment.json", {"python": sys.version, "executable": sys.executable,
         "platform": platform.platform(), "torch": torch.__version__, "torchvision": torchvision.__version__,
         "numpy": np.__version__, "pillow": PIL.__version__, "cuda_runtime": torch.version.cuda,
         "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(),
         "data_loader": {"workers": config["workers"], "prefetch_factor": config.get("loader_prefetch", 2),
                         "pin_memory": config.get("pin_memory", True)},
         "determinism": "strict" if config["deterministic"] else "seeded; cuDNN benchmark enabled; bitwise equality not guaranteed",
         "resume_note": "Epoch-boundary resume reseeds worker augmentation RNG by epoch; checkpoints include optimizer/scaler/RNG."})
    train_data = LeafDataset(ROOT / "data/images", manifests / "train.csv", config["image_size"], norm["mean"], norm["std"], True)
    val_data = LeafDataset(ROOT / "data/images", manifests / "val.csv", config["image_size"], norm["mean"], norm["std"])
    model = build_model(config["model"], len(classes)).to(device)
    dump(run / "model_profile.json", profile_model(model, config["image_size"]))
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"] == "fp16")
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    start_epoch, best_key = 0, (-1., -1.)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_key = ckpt["epoch"] + 1, tuple(ckpt["best_key"])
        random.setstate(ckpt["rng_python"])
        np.random.set_state(ckpt["rng_numpy"])
        torch.set_rng_state(ckpt["rng_torch"])
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        del ckpt
        gc.collect()
    stop_epoch = min(args.stop_after_epochs or config["epochs"], config["epochs"])
    print(f"GPU={torch.cuda.get_device_name()} train={len(train_data)} val={len(val_data)} start_epoch={start_epoch} stop_epoch={stop_epoch}", flush=True)
    fields = ["epoch", "learning_rate", "train_loss", "train_accuracy", "train_macro_f1", "val_loss", "val_accuracy", "val_macro_f1",
              "train_seconds", "val_seconds", "train_images_per_second", "peak_allocated_mib", "peak_reserved_mib"]
    history_path = run / "history.csv"
    if args.resume:
        with history_path.open(encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))
        if len(existing) != start_epoch or (existing and int(existing[-1]["epoch"]) != start_epoch):
            raise RuntimeError("History/checkpoint epoch mismatch; audit interrupted writes before resuming")
    session_start = time.perf_counter()
    completed_epochs = start_epoch
    stop_reason = "epoch_limit"
    for epoch in range(start_epoch, stop_epoch):
        if (ROOT / "STOP_TRAINING").exists():
            stop_reason = "stop_file"
            break
        if args.deadline_unix is not None and time.time() >= args.deadline_unix:
            stop_reason = "time_budget"
            break
        lr = lr_at(epoch, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        # New workers each epoch make worker RNG seed explicit at resume boundaries.
        train_generator = torch.Generator().manual_seed(config["seed"] + 1000 * epoch)
        val_generator = torch.Generator().manual_seed(config["seed"] + 1000 * epoch + 1)
        train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True,
                                  num_workers=config["workers"], pin_memory=config.get("pin_memory", True),
                                  prefetch_factor=config.get("loader_prefetch", 2) if config["workers"] else None,
                                  worker_init_fn=seed_worker, generator=train_generator)
        val_loader = DataLoader(val_data, batch_size=config["eval_batch_size"], shuffle=False,
                                num_workers=config["workers"], pin_memory=config.get("pin_memory", True),
                                prefetch_factor=config.get("loader_prefetch", 2) if config["workers"] else None,
                                worker_init_fn=seed_worker, generator=val_generator)
        torch.cuda.reset_peak_memory_stats()
        train_metrics = run_epoch(model, train_loader, criterion, device, config, optimizer, scaler, epoch)
        del train_loader
        val_metrics = run_epoch(model, val_loader, criterion, device, config, epoch=epoch)
        del val_loader
        record = {"epoch": epoch + 1, "learning_rate": lr,
                  **{f"train_{key}": train_metrics[key] for key in ("loss", "accuracy", "macro_f1", "seconds", "images_per_second")},
                  **{f"val_{key}": val_metrics[key] for key in ("loss", "accuracy", "macro_f1", "seconds")},
                  "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
        candidate = (val_metrics["macro_f1"], val_metrics["accuracy"])
        is_best = candidate > best_key
        if is_best:
            best_key = candidate
        checkpoint = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scaler": scaler.state_dict(), "config": config, "best_key": best_key,
                      "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                      "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
                      "val_metrics": val_metrics}
        atomic_save(checkpoint, run / "last.pt")
        if is_best:
            atomic_save(checkpoint, run / "best.pt")
            dump(run / "best_validation.json", {"epoch": epoch + 1, **val_metrics})
        with history_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(record)
        print(json.dumps(record), flush=True)
        completed_epochs = epoch + 1
    status = "training_complete" if completed_epochs == config["epochs"] else "stopped_at_epoch_boundary"
    summary = {"start_epoch": start_epoch + 1, "completed_epochs": completed_epochs,
         "planned_epochs": config["epochs"], "elapsed_session_seconds": time.perf_counter() - session_start,
         "best_val_macro_f1": best_key[0], "best_val_accuracy": best_key[1],
         "status": status, "stop_reason": stop_reason, "test_evaluated": False}
    dump(run / "session_summary.json", summary)
    sessions = run / "sessions"
    sessions.mkdir(exist_ok=True)
    dump(sessions / f"{time.time_ns()}.json", summary)


if __name__ == "__main__":
    main()
