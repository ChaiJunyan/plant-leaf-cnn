"""冻结并评价额外训练的检查点，支持复用已核验的历史逐图结果。"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from finish_supplemental import OUTPUT, evaluate, read, sha, write


def matching_existing_protocol(name, current):
    """历史结果保留原协议，逐项核对相同数据、精度与检查点。"""
    metric_path = OUTPUT / name / "test/clean/metrics.json"
    if not metric_path.exists():
        return None
    metric = read(metric_path)
    candidates = [ROOT / "configs/additional_evaluation_protocol.json",
                  ROOT / "configs/supplement_20261106/evaluation_protocol.json",
                  ROOT / "configs/supplement_20261106/evaluation_protocol_remaining.json"]
    for path in candidates:
        if not path.exists() or sha(path) != metric["protocol_sha256"]:
            continue
        previous = read(path)
        for key in ("seed", "batch_size", "classes", "precision", "test_manifest_sha256"):
            if previous[key] != current[key]:
                raise RuntimeError(f"Historical evaluation setting differs: {key}")
        if previous["checkpoint_sha256"].get(name) != current["checkpoint_sha256"][name]:
            raise RuntimeError("Historical checkpoint differs")
        for filename, digest in previous["source_sha256"].items():
            if sha(ROOT / filename) != digest:
                raise RuntimeError(f"Historical evaluator source changed: {filename}")
        return previous, path
    raise RuntimeError("Existing results have no matching frozen protocol")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="额外运行名称，可重复指定")
    args = parser.parse_args()
    if len(args.run) != len(set(args.run)):
        raise ValueError("Duplicate run names")
    classes = read(ROOT / "data/manifests/classes.json")
    sources = ("scripts/evaluate_additional.py", "scripts/finish_supplemental.py", "src/data.py",
               "src/models.py", "src/inference.py", "src/metrics.py")
    frozen = {"runs": args.run, "seed": 20261105, "batch_size": 128, "classes": classes,
              "precision": "float32", "cudnn_benchmark": False, "cudnn_deterministic": True,
              "test_manifest_sha256": sha(ROOT / "data/manifests/test.csv"),
              "checkpoint_sha256": {name: sha(ROOT / "runs" / name / "best.pt") for name in args.run},
              "source_sha256": {filename: sha(ROOT / filename) for filename in sources}}
    path = ROOT / "configs/additional_evaluation_protocol.json"
    if path.exists():
        protocol = read(path)
        if any(protocol.get(key) != value for key, value in frozen.items()):
            raise RuntimeError("An additional evaluation protocol already exists with different inputs")
    else:
        protocol = {**frozen, "frozen_utc": datetime.now(timezone.utc).isoformat()}
        write(path, protocol)
    for name in args.run:
        previous = matching_existing_protocol(name, protocol)
        if previous is None:
            evaluate(name, protocol, path)
        else:
            evaluate(name, previous[0], previous[1])
            print(f"Verified existing predictions: {name}", flush=True)


if __name__ == "__main__":
    main()
