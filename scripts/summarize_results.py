"""Summarize saved evaluation outputs without training, inference, or plotting."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("resnet18", "resnet18_ds", "resnet18_se", "resnet18_ds_se")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_summary(path, result):
    # Historical summaries are inputs to frozen protocols. Preserve their bytes.
    if path.exists():
        existing = read(path)
        for old, new in zip(existing["all_runs"], result["all_runs"]):
            for key in ("run", "accuracy", "macro_f1", "errors"):
                if old[key] != new[key]:
                    raise ValueError(f"Existing summary differs: {path}, {key}")
        if len(existing["all_runs"]) != len(result["all_runs"]):
            raise ValueError(f"Existing summary has a different run count: {path}")
        print(f"Verified existing summary: {path.relative_to(ROOT)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(f"Created summary: {path.relative_to(ROOT)}")


def aggregates(records, models):
    result = []
    for model in models:
        group = [r for r in records if r["model"] == model]
        result.append({"model": model, "n": len(group),
                       **{f"{metric}_{stat}": function([r[metric] for r in group])
                          for metric in ("accuracy", "macro_f1")
                          for stat, function in (("mean", statistics.mean), ("std", statistics.stdev))}})
    return result


def collect(name, directory):
    metric_path = directory / name / "test/clean/metrics.json"
    metric = read(metric_path)
    run = ROOT / "runs" / name
    profile = read(run / "model_profile.json")
    validation = read(run / "best_validation.json")
    history = list(csv.DictReader((run / "history.csv").open(encoding="utf-8", newline="")))
    if [int(row["epoch"]) for row in history] != list(range(1, 61)):
        raise ValueError(f"Expected 60 completed epochs: {name}")
    model, suffix = name.rsplit("_s", 1)
    return {"run": name, "model": model, "seed": int(suffix.split("_")[0]),
            "parameters": profile["parameters"], "macs": profile["conv_linear_macs"],
            "accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"], "errors": metric["errors"],
            "correct": metric["samples"] - metric["errors"], "checkpoint_epoch": validation["epoch"],
            "validation_macro_f1": validation["macro_f1"]}


def main():
    protocol = read(ROOT / "configs/final_evaluation_protocol.json")
    original = [collect(name, ROOT / "results/final") for name in protocol["runs"]]
    evidence = {}
    for record in original:
        name = record["run"]
        for filename in (f"results/final/{name}/test/clean/metrics.json", f"runs/{name}/history.csv",
                         f"runs/{name}/checkpoint_validation_check.json"):
            evidence[filename] = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
    final = {"main": [r for r in original if r["seed"] == 42], "all_runs": original,
             "repeats": aggregates(original, ("resnet18", "resnet18_ds_se")), "evidence_sha256": evidence,
             "labels_zh": read(ROOT / "assets/labels_zh.json"), "classes": protocol["classes"]}
    write_summary(ROOT / "results/final/summary.json", final)
    additional_names = ("resnet18_ds_s43_224", "resnet18_se_s43_224")
    additional_paths = [ROOT / "results/supplement_20261106" / name / "test/clean/metrics.json"
                        for name in additional_names]
    if not all(path.exists() for path in additional_paths):
        print("Main six-run summary is ready. Run again after both additional evaluations.")
        return
    records = original + [collect(name, ROOT / "results/supplement_20261106") for name in additional_names]
    pilots = []
    tuning_path = ROOT / "results/supplement_20261106/tuning_validation/summary.json"
    if tuning_path.exists():
        tuning = read(tuning_path)["evaluations"]
        for tag, smoothing in (("0", 0.0), ("005", 0.05)):
            name = f"resnet18_smoothing{tag}_s44_15of60"
            best = next(r for r in tuning if r["run"] == name and r["checkpoint"] == "best" and r["precision"] == "fp32")
            last = next(r for r in tuning if r["run"] == name and r["checkpoint"] == "last" and r["precision"] == "fp32")
            pilots.append({"run": name, "label_smoothing": smoothing, "completed_epochs": 15,
                           "lr_horizon": 60, "best_epoch": best["epoch"], "comparison_precision": "fp32",
                           "best_val_accuracy": best["metrics"]["accuracy"],
                           "best_val_macro_f1": best["metrics"]["macro_f1"],
                           "last_val_accuracy": last["metrics"]["accuracy"],
                           "last_val_macro_f1": last["metrics"]["macro_f1"], "test_evaluated": False})
    supplement = {"all_runs": records, "repeats": aggregates(records, MODELS), "validation_pilots": pilots,
                  "paired_pilots_complete": len(pilots) == 2, "post_hoc_supplement": True}
    write_summary(ROOT / "results/supplement_20261106/summary.json", supplement)
    print(json.dumps({"complete_structure_runs": len(records), "repeats": supplement["repeats"]}, indent=2))


if __name__ == "__main__":
    main()
