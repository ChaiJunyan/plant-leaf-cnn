"""对单张叶片图像进行分类，输出 top-5 分数、JSON 和识别结果图。

示例
    python predict.py --image assets/examples/example_0.jpg

程序复用训练时的模型结构与归一化统计，默认使用 CPU 和 FP32 推理。
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch

from src.inference import load_model, predict_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="待识别的叶片图片路径")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/leaf_ds.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "results/demo")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, config = load_model(args.checkpoint, args.device)
    labels = json.loads((ROOT / "assets/labels_zh.json").read_text(encoding="utf-8"))
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    # 预处理与前向计算由公共推理接口完成，避免单图与批量评价使用不同尺度。
    predictions = predict_image(model, image, config)
    result = {"image": str(args.image), "checkpoint": str(args.checkpoint),
              "model": config["model"], "device": args.device, "precision": "float32",
              "predictions": [{**row, "name": labels[row["label"]]} for row in predictions]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "prediction.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.family": ["Times New Roman", "SimSun"], "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.5), gridspec_kw={"width_ratios": [1, 1.15]})
    axes[0].imshow(image)
    axes[0].set_title("输入叶片图像")
    axes[0].axis("off")
    scores = [row["score"] * 100 for row in predictions]
    bars = axes[1].barh(range(len(scores)), scores, color=["#32745d"] + ["#a5bdb0"] * (len(scores) - 1))
    axes[1].set_yticks(range(len(scores)), [labels[row["label"]] for row in predictions])
    axes[1].invert_yaxis()
    axes[1].set(xlim=(0, 112), xlabel="softmax 分数 / %", title="top-5 分类结果")
    axes[1].bar_label(bars, labels=[f"{score:.2f}%" for score in scores], padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output / "prediction.png", dpi=280, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"模型 {config['model']}，设备 {args.device}，FP32")
    for rank, row in enumerate(result["predictions"], 1):
        print(f"{rank}. {row['name']}  {row['score']:.4%}")
    print(f"结果已保存到 {args.output.resolve()}")


if __name__ == "__main__":
    main()
