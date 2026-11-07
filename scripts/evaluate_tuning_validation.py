"""用统一FP32协议复核两组短预算调参，并诊断BF16临界分类差异。"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data import LeafDataset, seed_worker
from src.metrics import confusion_metrics
from src.models import build_model

OUTPUT = ROOT / "results/supplement_20261106/tuning_validation"
RUNS = ("resnet18_smoothing0_s44_15of60", "resnet18_smoothing005_s44_15of60")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate(name, checkpoint_name, precision, protocol):
    directory = OUTPUT / name / f"{checkpoint_name}_{precision}"
    directory.mkdir(parents=True, exist_ok=True)
    protocol_path = OUTPUT / "protocol.json"
    checkpoint_path = ROOT / "runs" / name / f"{checkpoint_name}.pt"
    if sha(checkpoint_path) != protocol["checkpoints"][name][checkpoint_name]:
        raise RuntimeError("Tuning checkpoint changed")
    metric_path = directory / "metrics.json"
    if metric_path.exists():
        saved = read(metric_path)
        if saved["protocol_sha256"] != sha(protocol_path):
            raise RuntimeError("Tuning validation protocol changed")
        for filename, digest in saved["output_sha256"].items():
            if sha(directory / filename) != digest:
                raise RuntimeError("Tuning output changed")
        return saved
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = precision == "bf16"
    torch.backends.cudnn.deterministic = precision == "fp32"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_model(config["model"], config["num_classes"]).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    norm = config["normalization"]
    dataset = LeafDataset(ROOT / "data/images", ROOT / "data/manifests/val.csv",
                          config["image_size"], norm["mean"], norm["std"])
    loader = DataLoader(dataset, batch_size=config["eval_batch_size"], num_workers=1,
                        prefetch_factor=1, pin_memory=False, worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(123))
    matrix = torch.zeros(config["num_classes"], config["num_classes"], dtype=torch.int64)
    rows, arrays = [], []
    stable = True
    begin = time.perf_counter()
    with torch.inference_mode():
        for step, (images, targets, indexes) in enumerate(loader, 1):
            images = images.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                logits = model(images)
            if precision == "fp32":
                second = model(images)
                stable = stable and torch.equal(logits, second)
            logits = logits.float().cpu()
            probabilities = logits.softmax(1)
            confidence, predicted = probabilities.max(1)
            top = logits.topk(2, dim=1).values
            matrix += torch.bincount(targets * config["num_classes"] + predicted,
                                     minlength=config["num_classes"]**2).reshape_as(matrix)
            arrays.append(logits.numpy())
            for index, target, prediction, score, margin in zip(indexes.tolist(), targets.tolist(),
                                                               predicted.tolist(), confidence.tolist(),
                                                               (top[:, 0]-top[:, 1]).tolist()):
                rows.append({"index": index, "path": dataset.rows[index]["path"], "target": target,
                             "predicted": prediction, "confidence": score, "logit_margin": margin,
                             "correct": int(target == prediction)})
            if step % 20 == 0:
                print(f"Validation {name} {checkpoint_name} {precision} batch={step}/{len(loader)}", flush=True)
    if precision == "fp32" and not stable:
        raise RuntimeError("Repeated deterministic FP32 forward differs")
    if [row["index"] for row in rows] != list(range(len(dataset))):
        raise RuntimeError("Validation sample order or count differs")
    independent = np.zeros(matrix.shape, dtype=np.int64)
    for row in rows:
        independent[row["target"], row["predicted"]] += 1
    assert np.array_equal(independent, matrix.numpy())
    metrics = confusion_metrics(matrix)
    f1 = (2*independent.diagonal()/np.maximum(independent.sum(0)+independent.sum(1), 1)).mean()
    assert abs(float(f1)-metrics["macro_f1"]) < 1e-12
    with (directory / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(directory / "logits.npz", logits=np.concatenate(arrays))
    saved_metrics = checkpoint["val_metrics"]
    result = {"run": name, "checkpoint": checkpoint_name, "epoch": checkpoint["epoch"]+1,
              "precision": precision, "validation_samples": len(rows), "test_evaluated": False,
              "checkpoint_sha256": sha(checkpoint_path), "protocol_sha256": sha(protocol_path),
              "independent_prediction_audit_passed": True, "fp32_repeated_forward_equal": stable if precision == "fp32" else None,
              "zero_logit_margin_samples": sum(row["logit_margin"] == 0 for row in rows),
              "metrics": metrics, "saved_training_validation_metrics": {k: saved_metrics[k] for k in ("accuracy", "macro_f1")},
              "difference_from_saved_training": {k: metrics[k]-saved_metrics[k] for k in ("accuracy", "macro_f1")},
              "confusion_matrix": matrix.tolist(), "elapsed_seconds": time.perf_counter()-begin,
              "output_sha256": {filename: sha(directory / filename) for filename in ("predictions.csv", "logits.npz")}}
    write(metric_path, result)
    print(json.dumps({"run": name, "checkpoint": checkpoint_name, "precision": precision,
                      "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                      "difference": result["difference_from_saved_training"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "protocol.json"
    frozen = {"runs": list(RUNS), "checkpoints": {name: {tag: sha(ROOT / "runs" / name / f"{tag}.pt")
                                                          for tag in ("best", "last")} for name in RUNS},
              "validation_manifest_sha256": sha(ROOT / "data/manifests/val.csv"),
              "sources": {filename: sha(ROOT / filename) for filename in
                          ("scripts/evaluate_tuning_validation.py", "src/data.py", "src/models.py", "src/metrics.py")},
              "evaluation": "Best checkpoint in training BF16 mode for diagnosis, then deterministic FP32 best/last for paired validation comparison. FP32 forwards repeated on each batch. No test evaluation."}
    if path.exists():
        protocol = read(path)
        if any(protocol.get(k) != v for k, v in frozen.items()):
            raise RuntimeError("Frozen validation protocol inputs changed")
    else:
        protocol = {**frozen, "frozen_utc": datetime.now(timezone.utc).isoformat()}
        write(path, protocol)
    results = []
    for name in RUNS:
        for checkpoint, precision in (("best", "bf16"), ("best", "fp32"), ("last", "fp32")):
            results.append(evaluate(name, checkpoint, precision, protocol))
    diagnosis = []
    for name in RUNS:
        bf = next(r for r in results if r["run"] == name and r["checkpoint"] == "best" and r["precision"] == "bf16")
        fp = next(r for r in results if r["run"] == name and r["checkpoint"] == "best" and r["precision"] == "fp32")
        with (OUTPUT / name / "best_bf16/predictions.csv").open(encoding="utf-8", newline="") as stream:
            bf_rows = list(csv.DictReader(stream))
        with (OUTPUT / name / "best_fp32/predictions.csv").open(encoding="utf-8", newline="") as stream:
            fp_rows = list(csv.DictReader(stream))
        differences = [{"index": b["index"], "target": b["target"], "bf16_prediction": b["predicted"],
                        "fp32_prediction": f["predicted"], "bf16_logit_margin": float(b["logit_margin"]),
                        "fp32_logit_margin": float(f["logit_margin"])}
                       for b, f in zip(bf_rows, fp_rows) if b["predicted"] != f["predicted"]]
        prior = ROOT / "runs" / name / "validation_predictions.csv"
        prior_differences = None
        if prior.exists():
            with prior.open(encoding="utf-8", newline="") as stream:
                previous = list(csv.DictReader(stream))
            if [r["path"] for r in previous] != [r["path"] for r in bf_rows]:
                raise RuntimeError("Prior validation sample paths differ")
            prior_differences = sum(p["predicted"] != b["predicted"] for p, b in zip(previous, bf_rows))
        diagnosis.append({"run": name, "bf16_vs_fp32_prediction_differences": differences,
                          "previous_bf16_vs_current_bf16_different_predictions": prior_differences,
                          "bf16_saved_metric_difference": bf["difference_from_saved_training"],
                          "fp32_repeat_stable": fp["fp32_repeated_forward_equal"]})
    write(OUTPUT / "summary.json", {"evaluations": results, "diagnosis": diagnosis,
                                    "protocol_sha256": sha(path), "test_evaluated": False,
                                    "comparison_precision": "fp32", "complete": True})


if __name__ == "__main__":
    main()
