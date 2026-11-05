"""按预先说明的四种正确性组合选择样本，比较基线与组合模型的 Grad-CAM。"""
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch

from src.inference import grad_cam, load_model, overlay_heatmap, predict_image


def main():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    labels = json.loads((ROOT / "assets/labels_zh.json").read_text(encoding="utf-8"))
    runs = ["resnet18_s42_224", "resnet18_ds_se_s42_224"]
    records = [list(csv.DictReader((ROOT / "results/final" / run / "test/clean/predictions.csv").open(encoding="utf-8"))) for run in runs]
    combinations = [(1, 1), (0, 1), (1, 0), (0, 0)]
    chosen = []
    for baseline_correct, modified_correct in combinations:
        pair = next((a, b) for a, b in zip(*records)
                    if int(a["correct"]) == baseline_correct and int(b["correct"]) == modified_correct)
        if pair[0]["index"] != pair[1]["index"]:
            raise AssertionError("Prediction order mismatch")
        chosen.append(pair)
    plt.rcParams.update({"font.family": ["Times New Roman", "SimSun"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(4, 3, figsize=(6.5, 8.6))
    metadata = []
    for row_index, pair in enumerate(chosen):
        with Image.open(ROOT / "data/images" / pair[0]["path"]) as source:
            image = source.convert("RGB")
        target = int(pair[0]["target"])
        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_title(f"真实类别\n{labels[target]}", fontsize=9)
        metadata.append({"index": int(pair[0]["index"]), "path": pair[0]["path"], "target": target,
                         "selection_rule": {"baseline_correct": combinations[row_index][0],
                                            "modified_correct": combinations[row_index][1]}, "models": {}})
    for column, run in enumerate(runs, 1):
        model, config = load_model(ROOT / "runs" / run / "best.pt", "cuda")
        for row_index, pair in enumerate(chosen):
            with Image.open(ROOT / "data/images" / pair[0]["path"]) as source:
                image = source.convert("RGB")
            prediction = predict_image(model, image, config, top_k=1)[0]
            if prediction["label"] != int(pair[column - 1]["predicted"]):
                raise AssertionError("Single-image prediction differs from saved FP32 evaluation")
            heat = grad_cam(model, image, config, prediction["label"])
            axes[row_index, column].imshow(overlay_heatmap(image, heat))
            title = "ResNet-18" if column == 1 else "DS+SE"
            axes[row_index, column].set_title(f'{title}\n{labels[prediction["label"]]} {prediction["score"]:.1%}', fontsize=9)
            metadata[row_index]["models"][run] = prediction
        del model
        torch.cuda.empty_cache()
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout(pad=0.8)
    output = ROOT / "results/final/figures"
    fig.savefig(output / "grad_cam_cases.png", dpi=280, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # 报告将两组正确性组合分别排版，保留同一幅原始拼图作为核对依据。
    with Image.open(output / "grad_cam_cases.png") as gallery:
        middle = gallery.height // 2
        gallery.crop((0, 0, gallery.width, middle)).save(output / "grad_cam_success.png")
        gallery.crop((0, middle, gallery.width, gallery.height)).save(output / "grad_cam_failure.png")
    (ROOT / "results/final/grad_cam_cases.json").write_text(json.dumps({
        "selection": "First test-manifest row in each (baseline correct, modified correct) combination: (1,1),(0,1),(1,0),(0,0). Same four images for both models.",
        "cases": metadata}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(chosen), "sample_indices": [row["index"] for row in metadata]}))


if __name__ == "__main__":
    main()
