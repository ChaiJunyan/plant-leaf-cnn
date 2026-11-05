"""Re-evaluate saved best checkpoint on validation and export predictions.

This first-stage verification intentionally has no test-set option.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader
from src.data import LeafDataset, seed_worker
from src.metrics import confusion_metrics
from src.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="baseline_pilot_s42_224")
    args = parser.parse_args()
    run = ROOT / "runs" / args.run
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    loader_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    torch.set_num_threads(config["cpu_threads"])
    # 与训练入口使用相同的卷积算法选择设置，减少 BF16 临界预测差异。
    torch.backends.cudnn.benchmark = not config["deterministic"]
    torch.backends.cudnn.deterministic = config["deterministic"]
    manifest = ROOT / "data/manifests/val.csv"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == config["manifest_sha256"]["val"]
    dataset = LeafDataset(ROOT / "data/images", manifest, config["image_size"],
                          config["normalization"]["mean"], config["normalization"]["std"])
    loader = DataLoader(dataset, batch_size=config["eval_batch_size"], num_workers=loader_config["workers"],
                        pin_memory=loader_config.get("pin_memory", True), worker_init_fn=seed_worker,
                        prefetch_factor=loader_config.get("loader_prefetch", 2) if loader_config["workers"] else None,
                        generator=torch.Generator().manual_seed(123))
    model = build_model(config["model"], config["num_classes"]).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    confusion = torch.zeros(config["num_classes"], config["num_classes"], dtype=torch.long, device="cuda")
    predictions = []
    dtype = torch.bfloat16 if config["amp"] == "bf16" else torch.float16
    begin = time.perf_counter()
    with torch.no_grad():
        for x, y, indexes in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=dtype, enabled=config["amp"] != "off"):
                logits = model(x)
            confidence, predicted = logits.float().softmax(1).max(1)
            confusion += torch.bincount(y * config["num_classes"] + predicted,
                                         minlength=config["num_classes"] ** 2).reshape_as(confusion)
            for idx, target, pred, prob in zip(indexes.tolist(), y.tolist(), predicted.tolist(), confidence.tolist()):
                row = dataset.rows[idx]
                predictions.append({"path": row["path"], "target": target, "predicted": pred,
                                    "confidence": prob, "correct": int(target == pred)})
    metrics = confusion_metrics(confusion)
    saved = checkpoint["val_metrics"]
    differences = {key: abs(metrics[key] - saved[key]) for key in ("accuracy", "macro_f1")}
    result = {"checkpoint_epoch": checkpoint["epoch"] + 1, "validation_samples": len(predictions),
              "metrics": metrics, "absolute_difference_from_saved_metrics": differences,
              "confusion_matrix": confusion.cpu().tolist(),
              "evaluation_loader": {key: loader_config.get(key) for key in ("workers", "loader_prefetch", "pin_memory")},
              "elapsed_seconds": time.perf_counter() - begin, "test_evaluated": False}
    result["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    result["matches_saved_metrics"] = all(delta <= 1e-12 for delta in differences.values())
    (run / "checkpoint_validation_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (run / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "target", "predicted", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(predictions)
    if not result["matches_saved_metrics"]:
        raise AssertionError(f"Checkpoint validation metrics differ from training log: {differences}; diagnostics saved")
    print(json.dumps({"epoch": result["checkpoint_epoch"], "accuracy": metrics["accuracy"],
                      "macro_f1": metrics["macro_f1"], "metric_differences": differences,
                      "test_evaluated": False}), flush=True)


if __name__ == "__main__":
    main()
