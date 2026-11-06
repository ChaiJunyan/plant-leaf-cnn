"""从真实预测和训练日志汇总结果，生成报告图表及可部署权重。"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image
import torch

MODELS = ["resnet18", "resnet18_ds", "resnet18_se", "resnet18_ds_se"]
NAMES = ["ResNet-18", "DS", "SE", "DS+SE"]
COLORS = ["#3d6487", "#32745d", "#ad704a", "#8a6e9b"]
OUT = ROOT / "results/final"
FIGURES = OUT / "figures"
plt.rcParams.update({"font.family": ["Times New Roman", "SimSun"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.unicode_minus": False, "savefig.dpi": 280})


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_figure(fig, name):
    """同时保留报告插图和可缩放矢量版本。"""
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def summarize_results():
    """以指标文件为唯一结果来源，汇总单种子主表和两次重复的描述统计。"""
    protocol = read_json(ROOT / "configs/final_evaluation_protocol.json")
    latency = read_json(OUT / "latency.json")
    if not latency.get("complete", False):
        raise RuntimeError("Finish all latency measurement rounds first")
    audit = read_json(ROOT / "data/manifests/audit.json")
    labels = read_json(ROOT / "assets/labels_zh.json")
    main, all_runs, evidence = [], [], {}
    for run in protocol["runs"]:
        metric_path = OUT / run / "test/clean/metrics.json"
        metrics = read_json(metric_path)
        profile = read_json(ROOT / "runs" / run / "model_profile.json")
        validation = read_json(ROOT / "runs" / run / "best_validation.json")
        check = read_json(ROOT / "runs" / run / "checkpoint_validation_check.json")
        if any(check["absolute_difference_from_saved_metrics"].values()):
            raise RuntimeError(f"Unresolved checkpoint verification: {run}")
        record = {"run": run, "model": run.rsplit("_s", 1)[0], "seed": int(run.rsplit("_s", 1)[1].split("_")[0]),
                  "parameters": profile["parameters"], "macs": profile["conv_linear_macs"],
                  "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                  "correct": metrics["correct"], "errors": metrics["errors"],
                  "checkpoint_epoch": validation["epoch"], "validation_macro_f1": validation["macro_f1"]}
        all_runs.append(dict(record))
        if record["seed"] == 42:
            measured = latency["runs"][run]
            stats = ("median_ms", "p05_ms", "p95_ms", "mean_ms", "std_ms")
            record["latency"] = {"cpu_batch1": {key: measured["cpu_batch1"][key] for key in stats},
                **{case: {scope: {key: measured[case][scope][key] for key in stats}
                          for scope in ("cuda_event", "synchronized_wall")}
                   for case in ("gpu_batch1", "gpu_batch32")},
                "round_medians": measured["round_medians"]}
            main.append(record)
        for path in (metric_path, ROOT / "runs" / run / "history.csv",
                     ROOT / "runs" / run / "checkpoint_validation_check.json"):
            evidence[str(path.relative_to(ROOT))] = fingerprint(path)
    repeats = []
    for model in ("resnet18", "resnet18_ds_se"):
        values = [row for row in all_runs if row["model"] == model]
        repeats.append({"model": model, "n": len(values),
                        **{f"{metric}_{stat}": function([row[metric] for row in values])
                           for metric in ("accuracy", "macro_f1")
                           for stat, function in (("mean", statistics.mean), ("std", statistics.stdev))}})
    baseline = main[0]
    for row in main:
        row["parameter_reduction"] = 1 - row["parameters"] / baseline["parameters"]
        row["mac_reduction"] = 1 - row["macs"] / baseline["macs"]
    smoothing = 0.05
    correct_target = 1 - smoothing + smoothing / len(labels)
    other_target = smoothing / len(labels)
    loss_floor = -correct_target * math.log(correct_target) - (len(labels) - 1) * other_target * math.log(other_target)
    result = {"main": main, "all_runs": all_runs, "repeats": repeats,
              "training_config": read_json(ROOT / "runs/resnet18_s42_224/config.json"),
              "dataset": {"total": audit["total_decoded"], "classes": audit["classes"],
                          "crops": len({name.split("___")[0] for name in protocol["classes"]}),
                          "counts": audit["counts"], "class_counts": audit["class_counts"],
                          "original_crossing_groups": len(audit["original_cross_split_connected_groups"]),
                          "fallback_leaf_images": audit["fallback_leaf_images"],
                          "extra_identical_images": audit["duplicate_pixel_images_beyond_first"],
                          "near_duplicate_pairs": audit["near_duplicate_candidates"]},
              "labels_zh": labels, "classes": protocol["classes"], "robustness_samples": protocol["robustness_samples"],
              "label_smoothing_entropy_floor": loss_floor, "evidence_sha256": evidence,
              "evaluation_protocol_sha256": fingerprint(ROOT / "configs/final_evaluation_protocol.json"),
              "repository_url": "https://github.com/ChaiJunyan/plant-leaf-cnn"}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = ["model", "seed", "parameters", "macs", "accuracy", "macro_f1", "correct", "errors", "checkpoint_epoch"]
    with (OUT / "main_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(main)
    return result, protocol


def data_figures(summary):
    classes = summary["classes"]
    fig, ax = plt.subplots(figsize=(7, 2.3))
    bottom = np.zeros(len(classes))
    for split, label, color in (("train", "训练集", "#32745d"), ("val", "验证集", "#8bb9ad"), ("test", "测试集", "#d7aa76")):
        values = np.array([summary["dataset"]["class_counts"][split].get(name, 0) for name in classes])
        ax.bar(np.arange(len(classes)), values, bottom=bottom, color=color, label=label, width=0.82)
        bottom += values
    ax.set(xlabel="类别编号", ylabel="图像数量")
    ax.set_xticks(range(38), range(38), fontsize=7)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    save_figure(fig, "class_distribution")
    rows = list(csv.DictReader((ROOT / "data/manifests/train.csv").open(encoding="utf-8")))
    fig, axes = plt.subplots(2, 6, figsize=(7.2, 3.8))
    for ax, label in zip(axes.flat, [0, 1, 3, 7, 8, 9, 11, 12, 20, 21, 29, 30]):
        row = next(row for row in rows if int(row["label"]) == label)
        with Image.open(ROOT / "data/images" / row["path"]) as image:
            ax.imshow(image.convert("RGB"))
        name = summary["labels_zh"][label]
        crop_length = 3 if name.startswith("马铃薯") else 2
        ax.set_title(f'{label} {name[:crop_length]}\n{name[crop_length:]}', fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, "training_examples")


def box(ax, x, y, width, height, text, color="#e7f0eb"):
    """绘制结构示意图中的一个计算模块。"""
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.01,rounding_size=0.02",
                               edgecolor="#597467", facecolor=color, linewidth=1))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)


def arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#45584f", "lw": 1.2})


def architecture_figures():
    fig, ax = plt.subplots(figsize=(7.3, 2.55))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    labels = ["输入\n224×224×3", "7×7 卷积\n池化", "layer1\n56×56×64", "layer2\n28×28×128", "layer3\n14×14×256", "layer4\n7×7×512", "全局均值\nFC → 38"]
    for i, label in enumerate(labels):
        x = 0.015 + i * 0.142
        box(ax, x, 0.48, 0.118, 0.35, label, "#dbe9e0" if i in (4, 5) else "#f3f5f0")
        if i < 6:
            arrow(ax, (x + 0.12, 0.655), (x + 0.14, 0.655))
    ax.text(0.5, 0.28, "每个 stage 含 2 个残差块；DS 替换 layer3 / layer4 分支中的 3×3 卷积", ha="center", fontsize=9)
    ax.text(0.5, 0.10, "SE 放在各残差块的末端 BN 之后、残差相加之前", ha="center", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "architecture")
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.02, 0.91, "（a）卷积分解", fontsize=11, weight="bold")
    box(ax, 0.04, 0.60, 0.16, 0.18, "输入\nCin 个通道")
    box(ax, 0.30, 0.60, 0.25, 0.18, "DW 3×3\n逐通道空间卷积")
    box(ax, 0.65, 0.60, 0.29, 0.18, "PW 1×1\n跨通道线性组合")
    arrow(ax, (0.20, 0.69), (0.30, 0.69))
    arrow(ax, (0.55, 0.69), (0.65, 0.69))
    ax.text(0.5, 0.50, "分解内部保持线性，外部沿用残差块原有 BN 与 ReLU", ha="center", fontsize=9)
    ax.text(0.02, 0.37, "（b）SE 通道重标定", fontsize=11, weight="bold")
    for x, width, text in [(0.04, 0.18, "全局均值\nC → C×1×1"), (0.31, 0.27, "降维 / ReLU / 升维\nC → C/16 → C"), (0.68, 0.26, "Sigmoid 权重\n逐通道相乘")]:
        box(ax, x, 0.10, width, 0.20, text)
    arrow(ax, (0.22, 0.20), (0.31, 0.20))
    arrow(ax, (0.58, 0.20), (0.68, 0.20))
    fig.tight_layout()
    save_figure(fig, "components")


def result_figures(summary):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for model, name, color in zip(MODELS, NAMES, COLORS):
        rows = list(csv.DictReader((ROOT / "runs" / f"{model}_s42_224/history.csv").open(encoding="utf-8")))
        epochs = np.array([int(row["epoch"]) for row in rows])
        train = np.array([float(row["train_loss"]) for row in rows])
        val = np.array([float(row["val_loss"]) for row in rows])
        f1 = np.array([float(row["val_macro_f1"]) * 100 for row in rows])
        axes[0, 0].plot(epochs, train, color=color, label=name, lw=1.6)
        axes[0, 1].plot(epochs, f1, color=color, lw=1.6)
        axes[1, 0].plot(epochs[9:], val[9:], color=color, lw=1.6)
        axes[1, 1].plot(epochs[9:], f1[9:], color=color, lw=1.6)
    for ax, title, ylabel in zip(axes.flat, ["（a）训练损失", "（b）验证集 Macro-F1", "（c）第10轮起的验证损失", "（d）第10轮起的验证集 Macro-F1"], ["交叉熵", "Macro-F1 / %", "交叉熵", "Macro-F1 / %"]):
        ax.set(title=title, xlabel="epoch", ylabel=ylabel)
        ax.grid(alpha=0.18)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "training_curves")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.25))
    for row, name, color in zip(summary["main"], NAMES, COLORS):
        x, y = row["parameters"] / 1e6, row["accuracy"] * 100
        axes[0].scatter(x, y, s=row["macs"] / 1e7, color=color, alpha=0.85)
        axes[0].annotate(name, (x, y), xytext=(4, -14 if name == "DS+SE" else 8), textcoords="offset points", fontsize=9)
        cpu = row["latency"]["cpu_batch1"]["median_ms"]
        gpu = row["latency"]["gpu_batch1"]["synchronized_wall"]["median_ms"]
        axes[1].scatter(cpu, gpu, s=80, color=color)
        axes[1].annotate(name, (cpu, gpu), xytext=(5, 5), textcoords="offset points", fontsize=9)
    axes[0].set(xlabel="参数量 / 百万", ylabel="测试准确率 / %", ylim=(99.3, 100.0), xlim=(0, 13))
    axes[0].text(0.02, 0.02, "气泡面积与 MACs 成正比", transform=axes[0].transAxes, fontsize=8)
    axes[1].set(xlabel="CPU 单图延迟 / ms", ylabel="GPU 单图同步墙钟延迟 / ms")
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.tight_layout()
    save_figure(fig, "efficiency_tradeoff")
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.8))
    for ax, model, name in zip(axes.flat, MODELS, NAMES):
        d = read_json(OUT / f"{model}_s42_224/test/clean/metrics.json")
        matrix = np.asarray(d["confusion_matrix"], dtype=float)
        image = ax.imshow(matrix / matrix.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        ax.set(title=name, xlabel="预测类别编号", ylabel="真实类别编号")
        ax.set_xticks(range(0, 38, 5))
        ax.set_yticks(range(0, 38, 5))
    fig.subplots_adjust(wspace=0.45, hspace=0.65, right=0.83)
    color_axis = fig.add_axes([0.88, 0.16, 0.025, 0.70])
    fig.colorbar(image, cax=color_axis, label="按真实类别归一化")
    save_figure(fig, "confusion_matrices")
    metrics = [read_json(OUT / f"{model}_s42_224/test/clean/metrics.json") for model in MODELS]
    values = np.array([d["per_class_f1"] for d in metrics]).T * 100
    selected = np.argsort(values.mean(1))[:10]
    fig, ax = plt.subplots(figsize=(6.7, 3.4))
    positions = np.arange(len(selected))
    ax.barh(positions - 0.17, values[selected, 0], 0.32, label="ResNet-18", color=COLORS[0])
    ax.barh(positions + 0.17, values[selected, 3], 0.32, label="DS+SE", color=COLORS[3])
    ax.set_yticks(positions, [summary["labels_zh"][i] for i in selected], fontsize=9)
    ax.set(xlabel="测试集 F1 / %", xlim=(85, 100.5))
    ax.invert_yaxis()
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "class_f1_comparison")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    conditions = ["clean", "brightness_0.6", "noise_0.05", "blur_1.5"]
    for model, name, color in zip(MODELS, NAMES, COLORS):
        scores = [read_json(OUT / f"{model}_s42_224/robustness/{condition}/metrics.json")["macro_f1"] * 100 for condition in conditions]
        ax.plot(range(4), scores, "o-", color=color, label=name, lw=1.8)
    ax.set_xticks(range(4), ["原图", "亮度 ×0.6", "高斯噪声 σ=0.05", "高斯模糊 r=1.5"])
    ax.set(ylabel="Macro-F1 / %", ylim=(65, 101))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    fig.tight_layout()
    save_figure(fig, "robustness")


def export_demo(protocol):
    """导出验证集选定模型的纯推理权重，并保留来源指纹。"""
    selected = protocol["deployment_run"]
    checkpoint = torch.load(ROOT / "runs" / selected / "best.pt", map_location="cpu", weights_only=False)
    destination = ROOT / "weights/leaf_ds_se.pt"
    destination.parent.mkdir(exist_ok=True)
    deployment = {"model": checkpoint["model"], "config": checkpoint["config"], "classes": protocol["classes"],
                  "source_checkpoint_sha256": protocol["checkpoint_sha256"][selected]}
    torch.save(deployment, destination)
    reloaded = torch.load(destination, map_location="cpu", weights_only=False)
    if not all(torch.equal(value, reloaded["model"][name]) for name, value in checkpoint["model"].items()):
        raise AssertionError("Exported inference weights changed")
    (ROOT / "weights/metadata.json").write_text(json.dumps({"run": selected, "bytes": destination.stat().st_size,
        "sha256": fingerprint(destination), "source_checkpoint_sha256": protocol["checkpoint_sha256"][selected]}, indent=2) + "\n", encoding="utf-8")
    examples = ROOT / "assets/examples"
    examples.mkdir(exist_ok=True)
    labels = read_json(ROOT / "assets/labels_zh.json")
    rows = list(csv.DictReader((ROOT / "data/manifests/val.csv").open(encoding="utf-8")))
    metadata = []
    for label in (0, 20, 30):
        row = next(row for row in rows if int(row["label"]) == label)
        name = f"example_{label}.jpg"
        with Image.open(ROOT / "data/images" / row["path"]) as image:
            image.convert("RGB").save(examples / name, quality=95)
        metadata.append({"name": labels[label], "file": name, "source_split": "val", "original_path": row["path"],
                         "source": "mohanty/PlantVillage", "license": "CC-BY-SA-3.0", "processing": "JPEG re-encoding for the interactive demo"})
    (examples / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary, protocol = summarize_results()
    data_figures(summary)
    architecture_figures()
    result_figures(summary)
    export_demo(protocol)
    print(json.dumps({"main_results": [{key: row[key] for key in ("model", "accuracy", "macro_f1", "parameters")} for row in summary["main"]],
                      "figures": len(list(FIGURES.glob('*.png'))), "loss_floor": summary["label_smoothing_entropy_floor"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
