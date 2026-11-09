"""在新增评价前固定子集、种子、模糊扫描、拒识校准及学习率对照。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
read = lambda path: json.loads(path.read_text(encoding="utf-8"))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
original = read(ROOT/"configs/final_evaluation_protocol.json")
models = ("resnet18", "resnet18_ds", "resnet18_se", "resnet18_ds_se")
runs = [f"{model}_s{seed}_224" for seed in (42, 43) for model in models]
sources = ("scripts/review_robustness.py", "scripts/review_lr_validation.py", "scripts/run_review_study.py",
           "scripts/evaluate_tuning_validation.py", "train.py", "evaluate.py", "src/data.py", "src/models.py",
           "src/metrics.py", "src/inference.py")
configs = ["configs/review_20261109/lr002_s44_15of60.json", "configs/review_20261109/lr01_s44_15of60.json"]
reference = ROOT/"runs/resnet18_smoothing005_s44_15of60"
reference_config = read(reference/"config.json")
for filename in configs:
    config = read(ROOT/filename)
    for key, value in config.items():
        if key not in ("learning_rate", "run_name"):
            assert reference_config[key] == value, key
protocol = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "budget_hours": 10,
            "runs": runs, "checkpoint_sha256": {name: sha(ROOT/"runs"/name/"best.pt") for name in runs},
            "source_sha256": {name: sha(ROOT/name) for name in sources},
            "training_configs": configs, "config_sha256": {name: sha(ROOT/name) for name in configs},
            "test_manifest_sha256": sha(ROOT/"data/manifests/test.csv"),
            "validation_manifest_sha256": sha(ROOT/"data/manifests/val.csv"),
            "robustness_indices": original["robustness_indices"], "noise_seed": original["noise_seed"],
            "conditions": ["clean", "brightness_0.6", "noise_0.05", "blur_0.5", "blur_1", "blur_1.5", "blur_2", "blur_2.5", "blur_3"],
            "calibration_runs": ["resnet18_ds_s42_224", "resnet18_ds_se_s42_224"],
            "rejection_quantile": 0.05,
            "rejection_policy": "Set each maximum-softmax threshold to lower fifth percentile on complete clean validation set before evaluating external images. Report known coverage and exploratory external acceptance; no external-label threshold fitting.",
            "lr_reference": {"run": reference.name, "learning_rate": 0.05,
                             "history_sha256": sha(reference/"history.csv"), "best_sha256": sha(reference/"best.pt"),
                             "last_sha256": sha(reference/"last.pt")},
            "lr_policy": "Learning rates0.02,0.05,0.1 share seed44, smoothing0.05, augmentation and first15of60 schedule. Reuse completed0.05 run. Validation-only post-hoc comparisons, best and epoch15 both reported.",
            "post_hoc_supplement": True, "original_evaluation_protocol_sha256": sha(ROOT/"configs/final_evaluation_protocol.json")}
destination = ROOT/"configs/review_20261109/protocol.json"
(ROOT/".cache/revision_20261109").mkdir(parents=True, exist_ok=True)
with destination.open("x", encoding="utf-8") as stream:
    json.dump(protocol, stream, indent=2, ensure_ascii=False)
    stream.write("\n")
print(json.dumps({"runs": len(runs), "robustness_cases": len(runs)*len(protocol["conditions"]),
                  "cached_original_cases": 16, "new_training_runs": len(configs)}))
