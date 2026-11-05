"""在首次测试前冻结检查点、推理精度及扰动子集，防止根据测试结果调参。"""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = ROOT / "configs/final_evaluation_protocol.json"
    if destination.exists():
        raise RuntimeError("Evaluation protocol already exists; inspect it instead of overwriting")
    queue = json.loads((ROOT / "runs/queue_status.json").read_text(encoding="utf-8"))
    if queue["status"] != "all_training_complete":
        raise RuntimeError("Complete the frozen training queue first")
    runs = queue["finished_runs"]
    for name in runs:
        run = ROOT / "runs" / name
        history = list(csv.DictReader((run / "history.csv").open(encoding="utf-8")))
        if len(history) != 60 or int(history[-1]["epoch"]) != 60:
            raise RuntimeError(f"Incomplete run: {name}")
    rows = list(csv.DictReader((ROOT / "data/manifests/test.csv").open(encoding="utf-8")))
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[int(row["label"])].append(index)
    rng = random.Random(20261105)
    selected = []
    for label in sorted(groups):
        selected.extend(rng.sample(groups[label], min(30, len(groups[label]))))
    source_names = ["evaluate.py", "src/inference.py", "src/data.py", "src/models.py", "src/metrics.py"]
    result = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "runs": runs,
              "classes": json.loads((ROOT / "data/manifests/classes.json").read_text(encoding="utf-8")),
              "test_samples": len(rows), "test_manifest_sha256": fingerprint(ROOT / "data/manifests/test.csv"),
              "checkpoint_sha256": {name: fingerprint(ROOT / "runs" / name / "best.pt") for name in runs},
              "evaluation_source_sha256": {name: fingerprint(ROOT / name) for name in source_names},
              "primary_metric": "macro_f1", "precision": "float32", "batch_size": 128,
              "seed": 20261105, "noise_seed": 20261105,
              "robustness_runs": [name for name in runs if "s42" in name],
              "conditions": ["clean", "brightness_0.6", "noise_0.05", "blur_1.5"],
              "robustness_indices": sorted(selected), "robustness_samples": len(selected),
              "robustness_selection": "30 test samples per class, fixed random seed, before observing predictions",
              "corruption_order": "RGB, bilinear resize to 224x224, fixed corruption, ToTensor, training normalization",
              "selection_policy": "Checkpoints and repeat architecture are fixed using training validation only. No test-guided training or checkpoint changes.",
              "deployment_run": "resnet18_ds_se_s42_224"}
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": runs, "test_samples": len(rows), "robustness_samples": len(selected), "precision": "float32"}))


if __name__ == "__main__":
    main()
