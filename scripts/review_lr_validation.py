"""对学习率0.02、0.05和0.1的15轮权重采用同一FP32验证协议。"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import evaluate_tuning_validation as validation
from review_robustness import read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    output = ROOT/"results/review_20261109/lr_validation"
    validation.OUTPUT = output
    names = ["resnet18_lr002_s44_15of60", "resnet18_lr01_s44_15of60"]
    for name in names:
        with (ROOT/"runs"/name/"history.csv").open(encoding="utf-8", newline="") as stream:
            import csv
            rows = list(csv.DictReader(stream))
        if [int(row["epoch"]) for row in rows] != list(range(1, 16)):
            raise RuntimeError(f"Incomplete learning-rate control: {name}")
    frozen = {"runs": names, "checkpoints": {name: {tag: sha(ROOT/"runs"/name/f"{tag}.pt")
                                                    for tag in ("best", "last")} for name in names},
              "validation_manifest_sha256": sha(ROOT/"data/manifests/val.csv"),
              "sources": {filename: sha(ROOT/filename) for filename in
                          ("scripts/review_lr_validation.py", "scripts/evaluate_tuning_validation.py",
                           "src/data.py", "src/models.py", "src/metrics.py")},
              "evaluation": "Deterministic FP32 best/last validation; fixed seed44, smoothing0.05 and first15of60 epochs; no test evaluation."}
    path = output/"protocol.json"
    if path.exists():
        protocol = read(path)
        if any(protocol.get(key) != value for key, value in frozen.items()):
            raise RuntimeError("Learning-rate validation protocol changed")
    else:
        protocol = {**frozen, "frozen_utc": datetime.now(timezone.utc).isoformat()}
        write(path, protocol)
    records = []
    for name in names:
        for tag in ("best", "last"):
            metric = validation.evaluate(name, tag, "fp32", protocol)
            records.append({"run": name, "checkpoint": tag, "epoch": metric["epoch"],
                            "accuracy": metric["metrics"]["accuracy"], "macro_f1": metric["metrics"]["macro_f1"],
                            "metrics_path": str((output/name/f"{tag}_fp32/metrics.json").relative_to(ROOT))})
    for tag in ("best", "last"):
        name = "resnet18_smoothing005_s44_15of60"
        path = ROOT/"results/supplement_20261106/tuning_validation"/name/f"{tag}_fp32/metrics.json"
        metric = read(path)
        if not metric["fp32_repeated_forward_equal"]:
            raise RuntimeError("Reference validation audit failed")
        records.append({"run": name, "checkpoint": tag, "epoch": metric["epoch"],
                        "accuracy": metric["metrics"]["accuracy"], "macro_f1": metric["metrics"]["macro_f1"],
                        "metrics_path": str(path.relative_to(ROOT))})
    write(output/"summary.json", {"complete": True, "records": records, "test_evaluated": False,
                                  "protocol_sha256": sha(output/"protocol.json")})


if __name__ == "__main__":
    main()
