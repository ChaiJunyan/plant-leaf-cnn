"""Recompute coursework metrics from the evidence archive. No images, GPU, or training required."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(read_bytes, names):
    def read(name):
        return json.loads(read_bytes(name).decode("utf-8-sig"))

    def rows(name):
        return list(csv.DictReader(io.StringIO(read_bytes(name).decode("utf-8-sig"))))

    protocol_names = [name for name in names if name.endswith(".json") and "protocol" in name]
    protocols = {hashlib.sha256(read_bytes(name)).hexdigest(): read(name) for name in protocol_names}
    manifests = {"test": rows("data/manifests/test.csv"), "val": rows("data/manifests/val.csv")}
    cases = sorted(name for name in names if name.endswith("/metrics.json")
                   and name.rsplit("/", 1)[0] + "/predictions.csv" in names)
    total = 0
    all_runs = []
    for name in cases:
        base = name.rsplit("/", 1)[0]
        saved = read(name)
        prediction = rows(base + "/predictions.csv")
        indices = np.array([int(r["index"]) for r in prediction])
        target = np.array([int(r["target"]) for r in prediction])
        predicted = np.array([int(r["predicted"]) for r in prediction])
        require(np.all((target >= 0) & (target < 38) & (predicted >= 0) & (predicted < 38)), name)
        matrix = np.bincount(target * 38 + predicted, minlength=38**2).reshape(38, 38)
        metric = saved.get("metrics", saved)
        accuracy = float((target == predicted).mean())
        f1 = 2 * matrix.diagonal() / np.maximum(matrix.sum(0) + matrix.sum(1), 1)
        require(matrix.tolist() == saved["confusion_matrix"], f"Confusion matrix: {name}")
        require(abs(accuracy - metric["accuracy"]) < 1e-12, f"Accuracy: {name}")
        require(abs(float(f1.mean()) - metric["macro_f1"]) < 1e-12, f"Macro-F1: {name}")
        if "per_class_f1" in metric:
            require(np.allclose(f1, metric["per_class_f1"], atol=1e-12, rtol=0), f"Per-class F1: {name}")
        per_class_path = base + "/per_class.csv"
        if per_class_path in names:
            per_class = rows(per_class_path)
            require([int(r["support"]) for r in per_class] == matrix.sum(1).tolist(), per_class_path)
            require(np.allclose([float(r["f1"]) for r in per_class], f1, atol=1e-12, rtol=0), per_class_path)
        require([int(r["correct"]) for r in prediction] == (target == predicted).astype(int).tolist(), name)
        protocol = protocols[saved["protocol_sha256"]]
        split = "val" if "validation_samples" in saved or "/calibration/" in name or "/lr_validation/" in name else "test"
        manifest = manifests[split]
        expected = protocol.get("robustness_indices") if saved.get("subset") else list(range(len(manifest)))
        require(indices.tolist() == expected, f"Sample order: {name}")
        require(all(r["path"] == manifest[int(r["index"])]["path"] and
                    r["target"] == manifest[int(r["index"])]["label"] for r in prediction), f"Manifest: {name}")
        for filename, digest in saved.get("output_sha256", {}).items():
            require(hashlib.sha256(read_bytes(base + "/" + filename)).hexdigest() == digest, filename)
        array_name = base + "/probabilities.npz"
        if array_name in names:
            with np.load(io.BytesIO(read_bytes(array_name)), allow_pickle=False) as arrays:
                probability = arrays["probabilities"]
                require(np.isfinite(probability).all() and (probability >= 0).all(), array_name)
                require(np.allclose(probability.sum(1), 1, atol=1e-6, rtol=0), array_name)
                require(np.array_equal(probability.argmax(1), predicted), array_name)
                require(np.array_equal(arrays["indices"], indices), array_name)
                require(np.allclose(probability.max(1), [float(r["confidence"]) for r in prediction], atol=1e-7, rtol=0), name)
        else:
            with np.load(io.BytesIO(read_bytes(base + "/logits.npz")), allow_pickle=False) as arrays:
                require(np.array_equal(arrays["logits"].argmax(1), predicted), name)
        total += len(prediction)
        if "/test/clean/" in name:
            run = saved["run"]
            model, suffix = run.rsplit("_s", 1)
            all_runs.append({"run": run, "model": model, "seed": int(suffix.split("_")[0]),
                             "accuracy": accuracy, "macro_f1": float(f1.mean()),
                             "errors": int((target != predicted).sum())})
    training = []
    for name in sorted(n for n in names if n.startswith("runs/resnet18") and n.endswith("/history.csv")):
        history = rows(name)
        expected_epochs = 15 if "15of60" in name else 60
        require([int(r["epoch"]) for r in history] == list(range(1, expected_epochs + 1)), name)
        best_path = name.rsplit("/", 1)[0] + "/best_validation.json"
        if best_path in names:
            best = read(best_path)
            selected = max(history, key=lambda r: (float(r["val_macro_f1"]), float(r["val_accuracy"])))
            require(best["epoch"] == int(selected["epoch"]), f"Best epoch: {name}")
            require(abs(best["macro_f1"] - float(selected["val_macro_f1"])) < 1e-12, name)
        training.append({"run": name.split("/")[1], "epochs": len(history)})
    repeats = []
    for model in ("resnet18", "resnet18_ds", "resnet18_se", "resnet18_ds_se"):
        group = [r for r in all_runs if r["model"] == model]
        if len(group) == 2:
            repeats.append({"model": model, "accuracy_mean_percent": statistics.mean(r["accuracy"] * 100 for r in group),
                            "accuracy_sample_std_pp": statistics.stdev(r["accuracy"] * 100 for r in group)})
    summary_name = "results/supplement_20261106/summary.json"
    if summary_name in names:
        summary = read(summary_name)
        for saved in summary["all_runs"]:
            actual = next(r for r in all_runs if r["run"] == saved["run"])
            for key in ("accuracy", "macro_f1", "errors"):
                require(abs(saved[key] - actual[key]) < 1e-12, f"Summary {saved['run']}: {key}")
        for saved in summary["repeats"]:
            group = [r for r in all_runs if r["model"] == saved["model"]]
            for key in ("accuracy", "macro_f1"):
                values = [r[key] for r in group]
                require(abs(statistics.mean(values) - saved[key + "_mean"]) < 1e-12, "Mean")
                require(abs(statistics.stdev(values) - saved[key + "_std"]) < 1e-12, "Sample std")
    external_count = 0
    external_name = "results/review_20261109/external_results.json"
    if external_name in names:
        external = read(external_name)
        with np.load(io.BytesIO(read_bytes("results/review_20261109/external_probabilities.npz")), allow_pickle=False) as arrays:
            probabilities = arrays["probabilities"]
            for probability, order in zip(probabilities, external["probability_order"]):
                case = next(c for c in external["cases"] if c["id"] == order["image"])
                image_path = "data/external_review_20261109/" + case["filename"]
                if image_path in names:
                    require(hashlib.sha256(read_bytes(image_path)).hexdigest() == case["sha256"], image_path)
                saved = case["models"][order["model"]]
                ranking = np.argsort(-probability)[:5]
                require(ranking.tolist() == [r["label"] for r in saved["top5"]], "External top-5")
                require([float(probability[i]) for i in ranking] == [r["score"] for r in saved["top5"]], "External scores")
                require(saved["accepted"] == (float(probability.max()) >= saved["threshold"]), "External acceptance")
                external_count += 1
        robust = read("results/review_20261109/robustness_summary.json")
        for calibration in robust["calibration"]:
            path = calibration["metrics_path"].replace("\\", "/").rsplit("/", 1)[0] + "/probabilities.npz"
            with np.load(io.BytesIO(read_bytes(path)), allow_pickle=False) as arrays:
                scores = arrays["probabilities"].max(1)
                threshold = float(np.quantile(scores, .05, method="lower"))
                require(threshold == calibration["threshold"], "Calibration threshold")
                require(float((scores >= threshold).mean()) == calibration["known_validation_coverage"], "Calibration coverage")
    latency_path = "results/final/latency.json"
    if latency_path in names:
        latency = read(latency_path)
        for record in latency["runs"].values():
            measurements = [record["cpu_batch1"]]
            for scope in ("gpu_batch1", "gpu_batch32"):
                measurements.extend(record[scope].values())
            for measurement in measurements:
                samples = np.array(measurement["samples"])
                require(abs(float(np.median(samples)) - measurement["median_ms"]) < 1e-12, "Latency median")
                require(abs(float(np.percentile(samples, 5)) - measurement["p05_ms"]) < 1e-12, "Latency P5")
                require(abs(float(np.percentile(samples, 95)) - measurement["p95_ms"]) < 1e-12, "Latency P95")
    return {"status": "passed", "evaluation_cases": len(cases), "prediction_rows": total,
            "external_predictions": external_count, "training_runs": training, "two_seed_results": repeats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path, default=ROOT / "evidence/coursework_evidence.zip")
    source.add_argument("--directory", type=Path, help="Check outputs generated by a new local reproduction")
    args = parser.parse_args()
    if args.directory:
        directory = args.directory.resolve()
        names = {p.relative_to(directory).as_posix() for folder in ("results", "configs", "data/manifests", "runs")
                 for p in (directory / folder).rglob("*") if p.is_file() and p.suffix in (".json", ".csv", ".npz")}
        result = verify(lambda name: (directory / name).read_bytes(), names)
    else:
        with zipfile.ZipFile(args.archive) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            names = set(manifest["files"])
            require(set(archive.namelist()) == names | {"manifest.json"}, "Archive entries")
            for name, digest in manifest["files"].items():
                require(hashlib.sha256(archive.read(name)).hexdigest() == digest, f"Archive SHA256: {name}")
            result = verify(archive.read, names)
            for key in ("evaluation_cases", "prediction_rows", "external_predictions"):
                require(result[key] == manifest["expected"][key], f"Evidence coverage: {key}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
