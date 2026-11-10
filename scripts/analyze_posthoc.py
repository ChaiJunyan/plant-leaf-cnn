"""Reproduce paired comparisons, calibration, and optional untrained stage profiles."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("resnet18", "resnet18_ds", "resnet18_se", "resnet18_ds_se")


def exact_mcnemar(gained, lost):
    discordant = gained + lost
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gained, lost) + 1)) / 2 ** discordant
    return min(1.0, 2 * tail)


def holm_adjust(records, field):
    ordered = sorted(range(len(records)), key=lambda i: records[i]["mcnemar_p"])
    previous = 0.0
    for rank, index in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - rank) * records[index]["mcnemar_p"]))
        records[index][field] = previous


def bootstrap_interval(difference, membership, repetitions, seed):
    """Multinomial draws over equal (group size, group difference) categories.

    For accuracy differences these two sufficient quantities reproduce the
    distribution of sampling entire groups, without a repetitions-by-images array.
    """
    sizes = np.bincount(membership)
    totals = np.bincount(membership, weights=difference).astype(np.int64)
    categories = Counter(zip(sizes.tolist(), totals.tolist()))
    values = np.array(list(categories), dtype=np.int64)
    probabilities = np.array(list(categories.values()), dtype=np.float64) / len(sizes)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(sizes), probabilities, size=repetitions)
    estimates = (draws @ values[:, 1]) / (draws @ values[:, 0])
    return (np.percentile(estimates, [2.5, 97.5]) * 100).tolist()


def analyze_archive(archive_path, repetitions, seed):
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))

        def payload(name):
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != manifest["files"][name]:
                raise ValueError(f"Evidence changed: {name}")
            return content

        def rows(name):
            return list(csv.DictReader(io.StringIO(payload(name).decode("utf-8-sig"))))

        test = rows("data/manifests/test.csv")
        groups, membership = np.unique([r["group_id"] for r in test], return_inverse=True)
        data = {}
        paths = {}
        for model in MODELS:
            for training_seed in (42, 43):
                run = f"{model}_s{training_seed}_224"
                folder = "supplement_20261106" if training_seed == 43 and model in ("resnet18_ds", "resnet18_se") else "final"
                base = f"results/{folder}/{run}/test/clean"
                prediction = rows(base + "/predictions.csv")
                assert [int(r["index"]) for r in prediction] == list(range(len(test)))
                assert all(r["target"] == test[i]["label"] and r["path"] == test[i]["path"] for i, r in enumerate(prediction))
                correct = np.array([int(r["target"]) == int(r["predicted"]) for r in prediction])
                confidence = np.array([float(r["confidence"]) for r in prediction])
                data[(model, training_seed)] = (correct, confidence)
                paths[run] = base
        pairs = []
        for training_seed in (42, 43):
            for left_index, reference in enumerate(MODELS):
                for model in MODELS[left_index + 1:]:
                    old, _ = data[(reference, training_seed)]
                    new, _ = data[(model, training_seed)]
                    gained = int((~old & new).sum())
                    lost = int((old & ~new).sum())
                    difference = new.astype(np.int64) - old.astype(np.int64)
                    pairs.append({"reference": reference, "model": model, "seed": training_seed,
                                  "reference_correct_model_wrong": lost,
                                  "reference_wrong_model_correct": gained,
                                  "accuracy_difference_pp": float(difference.mean() * 100),
                                  "mcnemar_p": exact_mcnemar(gained, lost),
                                  "image_bootstrap_ci95_pp": bootstrap_interval(difference, np.arange(len(test)), repetitions, seed),
                                  "group_bootstrap_ci95_pp": bootstrap_interval(difference, membership, repetitions, seed)})
        holm_adjust(pairs, "holm_all12_p")
        baseline = [r for r in pairs if r["reference"] == "resnet18"]
        holm_adjust(baseline, "holm_baseline6_p")
        calibration = []
        for model in MODELS:
            correct, confidence = data[(model, 42)]
            bins = np.clip(np.ceil(confidence * 15).astype(int) - 1, 0, 14)
            ece = 0.0
            bin_records = []
            for index in range(15):
                mask = bins == index
                count = int(mask.sum())
                if count:
                    accuracy = float(correct[mask].mean())
                    mean_score = float(confidence[mask].mean())
                    ece += count / len(test) * abs(accuracy - mean_score)
                    bin_records.append({"bin": index, "samples": count, "accuracy": accuracy, "confidence": mean_score})
            wrong = ~correct
            calibration.append({"model": model, "seed": 42, "accuracy_percent": float(correct.mean() * 100),
                                "mean_confidence_percent": float(confidence.mean() * 100), "ece15_percent": ece * 100,
                                "errors": int(wrong.sum()), "error_mean_confidence_percent": float(confidence[wrong].mean() * 100),
                                "error_max_confidence_percent": float(confidence[wrong].max() * 100), "bins": bin_records})
        latency = json.loads(payload("results/final/latency.json"))
        base = latency["runs"]["resnet18_s42_224"]
        ds = latency["runs"]["resnet18_ds_s42_224"]
        se = latency["runs"]["resnet18_se_s42_224"]
        timing = {"ds_cpu_reduction_percent": (1 - ds["cpu_batch1"]["median_ms"] / base["cpu_batch1"]["median_ms"]) * 100,
                  "ds_gpu_batch1_increase_percent": (ds["gpu_batch1"]["synchronized_wall"]["median_ms"] / base["gpu_batch1"]["synchronized_wall"]["median_ms"] - 1) * 100,
                  "ds_gpu_batch32_reduction_percent": (1 - ds["gpu_batch32"]["synchronized_wall"]["median_ms"] / base["gpu_batch32"]["synchronized_wall"]["median_ms"]) * 100,
                  "se_gpu_batch1_wall_increment_ms": se["gpu_batch1"]["synchronized_wall"]["median_ms"] - base["gpu_batch1"]["synchronized_wall"]["median_ms"],
                  "se_gpu_batch1_event_increment_ms": se["gpu_batch1"]["cuda_event"]["median_ms"] - base["gpu_batch1"]["cuda_event"]["median_ms"],
                  "se_gpu_batch32_wall_increment_ms": se["gpu_batch32"]["synchronized_wall"]["median_ms"] - base["gpu_batch32"]["synchronized_wall"]["median_ms"]}
        return {"samples": len(test), "known_association_groups": len(groups), "bootstrap_repetitions": repetitions,
                "bootstrap_seed": seed, "numpy_version": np.__version__, "difference_definition": "model minus reference",
                "mcnemar_scope": "Exploratory image-level exact tests; same-leaf views may be correlated.",
                "interval_scope": "Cluster bootstrap resamples known association groups for fixed checkpoints; does not estimate training-seed variation.",
                "baseline_comparisons": baseline, "all_pairwise": pairs, "calibration": calibration,
                "ece_definition": "15 equal-width confidence bins, right-closed; weighted absolute bin accuracy-confidence gaps.",
                "latency_derived": timing, "prediction_directories": paths}


def stage_profiles():
    sys.path.insert(0, str(ROOT))
    import torch
    from src.models import build_model, profile_model, separable

    torch.set_num_threads(4)
    records = []
    for stages in ((), (3, 4), (2, 3, 4), (1, 2, 3, 4)):
        torch.manual_seed(123)
        model = build_model("resnet18", 38)
        for number in stages:
            for block in getattr(model, f"layer{number}"):
                block.conv1 = separable(block.conv1)
                block.conv2 = separable(block.conv2)
        records.append({"ds_stages": list(stages), **profile_model(model, 224), "trained_for_this_comparison": False})
    for record in records:
        record["parameter_reduction_percent"] = (1 - record["parameters"] / records[0]["parameters"]) * 100
        record["mac_reduction_percent"] = (1 - record["conv_linear_macs"] / records[0]["conv_linear_macs"]) * 100
    return {"scope": "Untrained architecture profiles only; accuracy not measured for the extended replacements.", "profiles": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "evidence/coursework_evidence.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/posthoc_20261110")
    parser.add_argument("--repetitions", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--profile-stages", action="store_true", help="Also run untrained profiles, requires PyTorch")
    args = parser.parse_args()
    if args.repetitions < 1000:
        raise ValueError("Use at least 1000 bootstrap repetitions")
    args.output.mkdir(parents=True, exist_ok=True)
    result = analyze_archive(args.archive, args.repetitions, args.seed)
    result["evidence_archive_sha256"] = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    result["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output / "statistics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.profile_stages:
        profiles = stage_profiles()
        (args.output / "stage_profiles.json").write_text(json.dumps(profiles, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(profiles, indent=2))
    print(json.dumps({k: result[k] for k in ("samples", "known_association_groups", "baseline_comparisons", "calibration", "latency_derived")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
