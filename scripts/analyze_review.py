"""独立复算补充评价并绘制报告图，不更改训练或预测输出。"""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/review_20261109"
FIG = OUT / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

MODELS = ["resnet18", "resnet18_ds", "resnet18_se", "resnet18_ds_se"]
NAMES = ["ResNet-18", "DS", "SE", "DS+SE"]
COLORS = ["#3d6487", "#32745d", "#ad704a", "#8a6e9b"]
plt.rcParams.update({"font.family": ["Times New Roman", "SimSun"], "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.unicode_minus": False, "savefig.dpi": 240})


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def audit_case(path, expected_indices):
    """直接从保存的整数类别重建混淆矩阵，与概率及指标交叉检查。"""
    saved = read(path)
    prediction = rows(path.parent / "predictions.csv")
    assert [int(row["index"]) for row in prediction] == expected_indices
    target = np.array([int(row["target"]) for row in prediction])
    predicted = np.array([int(row["predicted"]) for row in prediction])
    assert [int(row["correct"]) for row in prediction] == (target == predicted).astype(int).tolist()
    matrix = np.bincount(target * 38 + predicted, minlength=38**2).reshape(38, 38)
    assert np.array_equal(matrix, saved["confusion_matrix"])
    metrics = saved.get("metrics", saved)
    f1 = 2 * matrix.diagonal() / np.maximum(matrix.sum(0) + matrix.sum(1), 1)
    assert abs(metrics["accuracy"] - (target == predicted).mean()) < 1e-12
    assert abs(metrics["macro_f1"] - f1.mean()) < 1e-12
    for filename, digest in saved["output_sha256"].items():
        assert sha(path.parent / filename) == digest
    if (path.parent / "probabilities.npz").exists():
        arrays = np.load(path.parent / "probabilities.npz")
        assert arrays["indices"].tolist() == expected_indices
        probability = arrays["probabilities"]
        assert np.array_equal(probability.argmax(1), predicted)
        assert np.all(np.isfinite(probability))
        assert np.max(np.abs(probability.sum(1) - 1)) < 1e-5
        assert np.max(np.abs(probability.max(1) - [float(row["confidence"]) for row in prediction])) < 1e-7
    else:
        logits = np.load(path.parent / "logits.npz")["logits"]
        assert np.array_equal(logits.argmax(1), predicted)
        assert saved["fp32_repeated_forward_equal"]
    return matrix, f1, len(prediction)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    protocol = read(ROOT / "configs/review_20261109/protocol.json")
    for collection in ("source_sha256", "config_sha256"):
        for filename, digest in protocol[collection].items():
            assert sha(ROOT / filename) == digest, filename
    for split in ("test", "validation"):
        filename = "val" if split == "validation" else split
        assert sha(ROOT / f"data/manifests/{filename}.csv") == protocol[f"{split}_manifest_sha256"]
    robust = read(OUT / "robustness_summary.json")
    assert robust["complete"] and len(robust["cases"]) == 72
    assert robust["protocol_sha256"] == sha(ROOT / "configs/review_20261109/protocol.json")
    count = 0
    balanced = []
    test_manifest = rows(ROOT / "data/manifests/test.csv")
    for case in robust["cases"]:
        path = ROOT / case["metrics_path"]
        assert sha(path) == case["metrics_sha256"]
        saved = read(path)
        assert saved["checkpoint_sha256"] == protocol["checkpoint_sha256"][case["run"]]
        matrix, f1, checked = audit_case(path, protocol["robustness_indices"])
        count += checked
        rr = rows(path.parent / "predictions.csv")
        assert all(r["path"] == test_manifest[int(r["index"])]["path"] and
                   r["target"] == test_manifest[int(r["index"])]["label"] for r in rr)
        assert np.all(matrix.sum(1) == 30)
        if case["condition"] == "clean" and "s42" in case["run"]:
            balanced.append({"run": case["run"], "errors": int(1140-matrix.trace()),
                             "macro_f1": float(f1.mean()),
                             "nonperfect_classes": [{"class": i, "tp": int(matrix[i, i]),
                               "support": int(matrix[i].sum()), "predicted": int(matrix[:, i].sum()),
                               "f1": float(f1[i])} for i in range(38) if f1[i] != 1]})
    calibration = []
    for case in robust["calibration"]:
        path = ROOT / case["metrics_path"]
        _, _, checked = audit_case(path, list(range(5498)))
        count += checked
        scores = np.load(path.parent / "probabilities.npz")["probabilities"].max(1)
        threshold = float(np.quantile(scores, .05, method="lower"))
        assert threshold == case["threshold"]
        assert float(np.mean(scores >= threshold)) == case["known_validation_coverage"]
        calibration.append({**case, "maximum_score": float(scores.max()),
                            "score_quantiles": np.quantile(scores, [0, .05, .5, .95, 1]).tolist()})
    original = read(ROOT / "results/final/summary.json")
    labels = original["labels_zh"]
    matrices, confusion = [], []
    for main in original["main"]:
        path = ROOT / f'results/final/{main["run"]}/test/clean/metrics.json'
        matrix, _, checked = audit_case(path, list(range(10709)))
        count += checked
        matrices.append(matrix)
        corn_errors = int(matrix[7, 9] + matrix[9, 7])
        confusion.append({"model": main["model"], "corn_pair_errors": corn_errors,
                          "total_errors": main["errors"], "fraction": corn_errors / main["errors"]})
    external = read(OUT / "external_results.json")
    assert external["complete"] and not external["disease_accuracy_estimated"]
    assert sha(OUT / "external_probabilities.npz") == external["probabilities_sha256"]
    extprotocol = read(OUT / "external_protocol.json")
    assert external["protocol_sha256"] == sha(OUT / "external_protocol.json")
    assert extprotocol["image_manifest_sha256"] == sha(ROOT / "assets/external_review_sources.json")
    for filename, digest in extprotocol["source_sha256"].items():
        assert sha(ROOT / filename) == digest
    for item in extprotocol["weights"]:
        assert sha(ROOT / "weights" / item["filename"]) == item["sha256"]
    extarray = np.load(OUT / "external_probabilities.npz")["probabilities"]
    for row, order in zip(extarray, external["probability_order"]):
        case = next(c for c in external["cases"] if c["id"] == order["image"])
        assert sha(ROOT / "data/external_review_20261109" / case["filename"]) == case["sha256"]
        record = case["models"][order["model"]]
        ranking = np.argsort(-row)[:5]
        assert ranking.tolist() == [item["label"] for item in record["top5"]]
        assert [float(row[i]) for i in ranking] == [item["score"] for item in record["top5"]]
        assert record["accepted"] == (float(row.max()) >= record["threshold"])
    # 用相同图像的两个种子画曲线，阴影为观测范围，不作置信区间解释。
    radii = [0, .5, 1, 1.5, 2, 2.5, 3]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    blur = []
    for model, name, color in zip(MODELS, NAMES, COLORS):
        values = []
        for seed in (42, 43):
            line = []
            for radius in radii:
                condition = "clean" if radius == 0 else f"blur_{radius:g}"
                case = next(c for c in robust["cases"] if c["run"] == f"{model}_s{seed}_224" and c["condition"] == condition)
                line.append(case["macro_f1"] * 100)
            values.append(line)
            blur.append({"model": model, "seed": seed, "radii": radii, "macro_f1_percent": line})
        array = np.array(values)
        ax.plot(radii, array.mean(0), "o-", color=color, label=name, ms=4)
        ax.fill_between(radii, array.min(0), array.max(0), color=color, alpha=.12)
    ax.set(xlabel="PIL 高斯模糊半径 / 像素（224×224 输入）", ylabel="Macro-F1 / %", ylim=(0, 102), xticks=radii)
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(.5, 1.02))
    ax.grid(alpha=.18)
    fig.tight_layout()
    save(fig, "blur_scan")
    repeats = read(ROOT / "results/supplement_20261106/summary.json")["repeats"]
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    for index, (main, repeat, name, color) in enumerate(zip(original["main"], repeats, NAMES, COLORS)):
        assert main["model"] == repeat["model"]
        x = main["parameters"] / 1e6
        y = repeat["accuracy_mean"] * 100
        ax.errorbar(x, y, yerr=repeat["accuracy_std"] * 100, fmt="o", color=color, capsize=5, ms=7, label=name)
        offset = [(12, -22), (12, -3), (12, 1), (12, -20)][index]
        ax.annotate(name, (x, y), xytext=offset, textcoords="offset points", color=color)
    ax.set(xlabel="模型参数量 / 百万", ylabel="测试准确率 / %", xlim=(.5, 13.5), ylim=(99.50, 99.82))
    ax.text(.98, .02, "纵轴局部显示；误差棒为两种子样本标准差", transform=ax.transAxes, ha="right", fontsize=10)
    ax.grid(alpha=.18)
    fig.tight_layout()
    save(fig, "parameters_accuracy")
    values = np.array([2*m.diagonal()/np.maximum(m.sum(0)+m.sum(1), 1) for m in matrices]).T * 100
    selected = np.argsort(values.mean(1))[:10]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for k, (name, color) in enumerate(zip(NAMES, COLORS)):
        ax.scatter(values[selected, k], np.arange(10)+(k-1.5)*.15, label=name, color=color, s=28, marker=["o", "s", "^", "D"][k])
    ax.set_yticks(range(10), [labels[i] for i in selected])
    ax.set(xlabel="逐类 F1 / %（横轴局部显示 90–100）", xlim=(90, 100.25))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.18)
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.01))
    fig.tight_layout()
    save(fig, "class_f1_points")
    fig, ax = plt.subplots(figsize=(7.5, 2.9))
    for tag, name, color in (("0", "ε = 0", COLORS[0]), ("005", "ε = 0.05", COLORS[1])):
        history = rows(ROOT / f"runs/resnet18_smoothing{tag}_s44_15of60/history.csv")
        ax.plot([int(r["epoch"]) for r in history], [float(r["val_macro_f1"])*100 for r in history], "o-", ms=3, label=name, color=color)
    ax.set(xlabel="epoch", ylabel="验证 Macro-F1 / %", ylim=(70, 100), xticks=range(1, 16, 2))
    ax.text(.98, .04, "局部纵轴 70–100；训练日志 BF16 值", transform=ax.transAxes, ha="right", fontsize=10)
    ax.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.01))
    ax.grid(alpha=.18)
    fig.tight_layout()
    save(fig, "smoothing_curves")
    # 网络图片仅显示来源描述和编号，模型 top-5 在可编辑表中列出。
    fig, axes = plt.subplots(2, 3, figsize=(8, 6.2))
    for ax, case in zip(axes.flat, external["cases"]):
        with Image.open(ROOT / "data/external_review_20261109" / case["filename"]) as source:
            image = source.convert("RGB")
            image.thumbnail((700, 700))
            ax.imshow(image)
        ax.set_title(f'{case["id"]}  {case["description_zh"]}', fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    save(fig, "external_examples")
    lrpath = OUT / "lr_validation/summary.json"
    lrrecords = []
    if lrpath.exists():
        lrsummary = read(lrpath)
        assert lrsummary["complete"] and not lrsummary["test_evaluated"]
        for record in lrsummary["records"]:
            _, _, checked = audit_case(ROOT / record["metrics_path"], list(range(5498)))
            count += checked
        fig, ax = plt.subplots(figsize=(7.5, 2.9))
        for rate, run, color in ((.02, "resnet18_lr002_s44_15of60", COLORS[0]),
                                  (.05, "resnet18_smoothing005_s44_15of60", COLORS[1]),
                                  (.1, "resnet18_lr01_s44_15of60", COLORS[2])):
            history = rows(ROOT / f"runs/{run}/history.csv")
            assert [int(r["epoch"]) for r in history] == list(range(1, 16))
            ax.plot(range(1, 16), [float(r["val_macro_f1"])*100 for r in history], "o-", ms=3, color=color, label=f"初始学习率 {rate:g}")
            best = next(r for r in lrsummary["records"] if r["run"] == run and r["checkpoint"] == "best")
            last = next(r for r in lrsummary["records"] if r["run"] == run and r["checkpoint"] == "last")
            lrrecords.append({"learning_rate": rate, "run": run, "best": best, "last": last})
        ax.set(xlabel="epoch", ylabel="验证 Macro-F1 / %", ylim=(70, 100), xticks=range(1, 16, 2))
        ax.text(.98, .04, "局部纵轴 70–100；训练日志 BF16 值", transform=ax.transAxes, ha="right", fontsize=10)
        ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.01))
        ax.grid(alpha=.18)
        fig.tight_layout()
        save(fig, "learning_rate_curves")
    output = {"audit_status": "passed", "robustness_cases": 72, "calibration_cases": 2,
              "external_cases": 12, "prediction_rows_checked": count, "learning_rate_complete": bool(lrrecords),
              "balanced_subset": balanced, "single_isolated_error_f1_drop_pp": (1/59+1/61)/38*100,
              "corn_confusions": confusion, "calibration": calibration, "blur_scan": blur,
              "smoothing_target_probability": 1-.05+.05/38, "learning_rates": lrrecords,
              "source_sha256": sha(Path(__file__)), "protocol_sha256": sha(ROOT / "configs/review_20261109/protocol.json")}
    (OUT / "analysis.json").write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("audit_status", "robustness_cases", "prediction_rows_checked", "learning_rate_complete", "corn_confusions")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
