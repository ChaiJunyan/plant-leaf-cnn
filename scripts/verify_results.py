"""独立复算公开预测结果并核对哈希，无需 GPU、原图或六组训练权重。"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_case(directory, protocol):
    """从 CSV 标签重新构造混淆矩阵，并与 NPZ 概率和已存指标交叉核对。"""
    metric = read_json(directory / "metrics.json")
    for name, expected in metric["output_sha256"].items():
        require(sha256(directory / name) == expected, f"Output hash: {directory / name}")
    require(metric["protocol_sha256"] == sha256(ROOT / "configs/final_evaluation_protocol.json"), "Protocol hash")
    require(metric["checkpoint_sha256"] == protocol["checkpoint_sha256"][metric["run"]], "Checkpoint provenance")
    with (directory / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    indices = np.array([int(row["index"]) for row in rows])
    targets = np.array([int(row["target"]) for row in rows])
    predicted = np.array([int(row["predicted"]) for row in rows])
    expected_indices = protocol["robustness_indices"] if metric["subset"] else list(range(protocol["test_samples"]))
    require(indices.tolist() == expected_indices, f"Sample order: {directory}")
    with np.load(directory / "probabilities.npz", allow_pickle=False) as archive:
        probabilities = archive["probabilities"]
        require(np.array_equal(indices, archive["indices"]), "NPZ index order")
        require(probabilities.shape == (len(rows), len(protocol["classes"])), "Probability shape")
        require(np.isfinite(probabilities).all() and (probabilities >= 0).all(), "Invalid probabilities")
        require(np.allclose(probabilities.sum(1), 1, atol=1e-6, rtol=0), "Probability sums")
        require(np.array_equal(probabilities.argmax(1), predicted), "NPZ/CSV predictions")
        require(np.allclose(probabilities.max(1), [float(row["confidence"]) for row in rows], atol=0, rtol=0), "Confidence mismatch")
    classes = len(protocol["classes"])
    confusion = np.bincount(targets * classes + predicted, minlength=classes**2).reshape(classes, classes)
    require(confusion.tolist() == metric["confusion_matrix"], "Confusion matrix")
    diagonal = confusion.diagonal().astype(np.float64)
    support, outputs = confusion.sum(1), confusion.sum(0)
    precision = diagonal / np.maximum(outputs, 1)
    recall = diagonal / np.maximum(support, 1)
    f1 = 2 * diagonal / np.maximum(support + outputs, 1)
    for name, value in (("accuracy", diagonal.sum() / len(rows)), ("macro_f1", f1.mean()),
                        ("per_class_precision", precision), ("per_class_recall", recall), ("per_class_f1", f1)):
        require(np.allclose(value, metric[name], atol=1e-12, rtol=0), f"Recomputed {name}: {directory}")
    require(metric["samples"] == len(rows) and metric["errors"] == int((targets != predicted).sum()), "Sample/error count")
    require([int(row["correct"]) for row in rows] == (targets == predicted).astype(int).tolist(), "Correctness flags")
    manifest = list(csv.DictReader((ROOT / "data/manifests/test.csv").open(encoding="utf-8")))
    require(all(row["path"] == manifest[int(row["index"])]["path"] and row["target"] == manifest[int(row["index"])]["label"] for row in rows), "Test manifest alignment")
    per_class = list(csv.DictReader((directory / "per_class.csv").open(encoding="utf-8")))
    require(len(per_class) == classes, "Per-class count")
    for i, row in enumerate(per_class):
        require(int(row["support"]) == support[i] and row["class_name"] == protocol["classes"][i], "Per-class metadata")
        for key, values in (("precision", precision), ("recall", recall), ("f1", f1)):
            require(abs(float(row[key]) - values[i]) <= 1e-12, f"Per-class {key}")
    return metric


def main():
    protocol = read_json(ROOT / "configs/final_evaluation_protocol.json")
    for name, expected in protocol["evaluation_source_sha256"].items():
        require(sha256(ROOT / name) == expected, f"Frozen source changed: {name}")
    require(sha256(ROOT / "data/manifests/test.csv") == protocol["test_manifest_sha256"], "Test manifest hash")
    cases = []
    for run in protocol["runs"]:
        cases.append(verify_case(ROOT / "results/final" / run / "test/clean", protocol))
    for run in protocol["robustness_runs"]:
        for condition in protocol["conditions"]:
            cases.append(verify_case(ROOT / "results/final" / run / "robustness" / condition, protocol))
    summary = read_json(ROOT / "results/final/summary.json")
    for row in summary["all_runs"]:
        metric = next(case for case in cases if case["run"] == row["run"] and not case["subset"])
        require(row["accuracy"] == metric["accuracy"] and row["macro_f1"] == metric["macro_f1"], "Summary differs")
    for name, expected in summary["evidence_sha256"].items():
        path = ROOT / name
        if not path.exists() and name.startswith("runs/"):
            path = ROOT / "records/training" / name[len("runs/"):]
        require(sha256(path) == expected, f"Summary evidence: {name}")
    for row in summary["repeats"]:
        repeated = [item for item in summary["all_runs"] if item["model"] == row["model"]]
        for key in ("accuracy", "macro_f1"):
            values = [item[key] for item in repeated]
            require(abs(np.mean(values) - row[key + "_mean"]) < 1e-12, "Repeat mean")
            require(abs(np.std(values, ddof=1) - row[key + "_std"]) < 1e-12, "Repeat sample std")
    print(json.dumps({"status": "passed", "evaluation_cases": len(cases),
                      "prediction_rows": sum(case["samples"] for case in cases),
                      "test_runs": len(protocol["runs"]), "precision": "float32"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
