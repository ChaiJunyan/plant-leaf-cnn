"""从已保存的逐图输出生成报告修订分析，不重新训练或改写冻结结果。"""
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/revision_20261106"
FIG = OUT / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image

MODELS = ["resnet18", "resnet18_ds", "resnet18_se", "resnet18_ds_se"]
NAMES = ["ResNet-18", "DS", "SE", "DS+SE"]
COLORS = ["#3d6487", "#32745d", "#ad704a", "#8a6e9b"]
plt.rcParams.update({"font.family": ["Times New Roman", "SimSun"], "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.unicode_minus": False, "savefig.dpi": 260})


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


def box(ax, x, y, w, h, text, size=11, color="#e7f0eb"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.012",
                              edgecolor="#597467", facecolor=color, linewidth=1))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=size)


def arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "->", "color": "#45584f", "lw": 1.3})


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    summary = read(ROOT / "results/final/summary.json")
    labels = summary["labels_zh"]
    evidence = {}
    validation, predictions, metrics = [], [], []
    for model in MODELS:
        run = f"{model}_s42_224"
        vp = ROOT / "runs" / run / "best_validation.json"
        v = read(vp)
        validation.append({"model": model, "epoch": v["epoch"], "accuracy": v["accuracy"],
                           "macro_f1": v["macro_f1"]})
        evidence[str(vp.relative_to(ROOT))] = sha(vp)
        base = ROOT / "results/final" / run
        predictions.append(rows(base / "test/clean/predictions.csv"))
        metrics.append(read(base / "test/clean/metrics.json"))
    # 验证同一子集上指标相同是否来自独立预测；保留概率、错误位置和检查点指纹。
    subset, subset_probability = [], []
    for model in MODELS[:3]:
        base = ROOT / "results/final" / f"{model}_s42_224/robustness/clean"
        rr = rows(base / "predictions.csv")
        pp = np.load(base / "probabilities.npz")
        prob = pp["probabilities"]
        assert np.array_equal(pp["indices"], [int(r["index"]) for r in rr])
        assert np.array_equal(prob.argmax(1), [int(r["predicted"]) for r in rr])
        matrix = np.zeros((38, 38), dtype=np.int64)
        for r in rr:
            matrix[int(r["target"]), int(r["predicted"])] += 1
        f1 = (2 * matrix.diagonal() / np.maximum(matrix.sum(0)+matrix.sum(1), 1)).mean()
        saved = read(base / "metrics.json")
        assert abs(f1 - saved["macro_f1"]) < 1e-12
        subset.append({"model": model, "macro_f1_recomputed": float(f1),
                       "errors": [{k: r[k] for k in ("index", "target", "predicted")} for r in rr if r["correct"] == "0"],
                       "checkpoint_sha256": saved["checkpoint_sha256"],
                       "probabilities_sha256": sha(base / "probabilities.npz"),
                       "predictions_sha256": sha(base / "predictions.csv")})
        subset_probability.append(prob)
    subset_comparison = []
    for i in range(3):
        for j in range(i+1, 3):
            a, b = subset_probability[i], subset_probability[j]
            subset_comparison.append({"models": [MODELS[i], MODELS[j]],
                                      "different_argmax": int(np.count_nonzero(a.argmax(1)!=b.argmax(1))),
                                      "max_probability_difference": float(np.max(np.abs(a-b))),
                                      "identical_probability_values": bool(np.array_equal(a,b))})
    matrices = np.array([m["confusion_matrix"] for m in metrics], dtype=np.int64)
    pair_counts = matrices.sum(0)
    np.fill_diagonal(pair_counts, 0)
    positions = sorted(zip(*np.nonzero(pair_counts)), key=lambda ij: (-pair_counts[ij], ij))[:10]
    pairs = [{"target": int(i), "predicted": int(j), "counts": matrices[:,i,j].tolist()}
             for i,j in positions]
    # 正确样例每个作物最多一张，取清单首张，避免只展示最高置信度图片。
    selected, crops = [], set()
    for index, (ds, combo) in enumerate(zip(predictions[1], predictions[3])):
        crop = summary["classes"][int(ds["target"])].split("___")[0]
        if ds["correct"] == combo["correct"] == "1" and crop not in crops:
            selected.append(index)
            crops.add(crop)
            if len(selected) == 6:
                break
    # 三类错误组合各取两张，限制重复作物，完整保留结果而非用其估计总体精度。
    for pattern in (("1", "0"), ("0", "1"), ("0", "0")):
        group, seen = [], set()
        for index, (ds, combo) in enumerate(zip(predictions[1], predictions[3])):
            crop = summary["classes"][int(ds["target"])].split("___")[0]
            if (ds["correct"], combo["correct"]) == pattern and crop not in seen:
                group.append(index)
                seen.add(crop)
                if len(group) == 2:
                    break
        selected.extend(group)
    cases = [{"index": i, "path": predictions[1][i]["path"],
              "target": int(predictions[1][i]["target"]),
              "ds": predictions[1][i], "combo": predictions[3][i]} for i in selected]
    result = {"validation_selection": validation, "subset_independence": subset,
              "subset_pairwise": subset_comparison, "common_confusions": pairs,
              "scenario_cases": cases, "evidence_sha256": evidence,
              "selection_policy": "Six first both-correct cases from distinct crops, followed by two distinct-crop cases for each DS/combo correctness pattern (1,0), (0,1), (0,0). Descriptive examples, not an unbiased accuracy estimate."}
    (OUT / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8.4, 2.8))
    bottom = np.zeros(38)
    for split, name, color in (("train", "训练集", "#32745d"), ("val", "验证集", "#8bb9ad"), ("test", "测试集", "#d7aa76")):
        values = np.array([summary["dataset"]["class_counts"][split].get(c,0) for c in summary["classes"]])
        ax.bar(range(38), values, bottom=bottom, color=color, label=name, width=.82)
        bottom += values
    ax.set(xlabel="类别编号", ylabel="图像数量")
    ax.set_xticks(range(38), range(38), fontsize=8)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5,1.01))
    fig.tight_layout()
    save(fig, "class_distribution")
    fig, ax = plt.subplots(figsize=(8.5,2.4))
    ax.set(xlim=(0,1), ylim=(0,1))
    ax.axis("off")
    texts = ["输入\n224×224×3", "7×7卷积\n池化", "layer1\n56×56×64", "layer2\n28×28×128", "layer3\n14×14×256", "layer4\n7×7×512", "全局均值\nFC→38"]
    for i,t in enumerate(texts):
        x=.005+i*.145
        box(ax,x,.50,.104,.29,t,size=9.8,color="#dbe9e0" if i in (4,5) else "#f3f5f0")
        if i < 6:
            arrow(ax,(x+.108,.645),(x+.141,.645))
    ax.text(.5,.29,"各阶段包含两个残差块；DS 仅替换 layer3 与 layer4 的分支卷积",ha="center",fontsize=11)
    ax.text(.5,.10,"SE 位于每个残差块的末端 BN 之后、残差相加之前",ha="center",fontsize=11)
    fig.tight_layout()
    save(fig,"architecture")
    fig,ax=plt.subplots(figsize=(8.0,4.3))
    ax.set(xlim=(0,1),ylim=(0,1))
    ax.axis("off")
    ax.text(.02,.94,"（a）卷积分解",fontsize=13,weight="bold")
    for x,w,t in ((.02,.18,"输入\nCin 个通道"),(.30,.25,"DW 3×3\n逐通道空间卷积"),(.66,.32,"PW 1×1\n跨通道线性组合")):
        box(ax,x,.66,w,.18,t,size=13)
    arrow(ax,(.205,.75),(.295,.75))
    arrow(ax,(.555,.75),(.655,.75))
    ax.text(.5,.54,"DW 与 PW 之间不加 BN/ReLU；外部沿用原残差块设置",ha="center",fontsize=11)
    ax.text(.02,.39,"（b）SE 通道重标定",fontsize=13,weight="bold")
    for x,w,t in ((.02,.22,"全局均值\nC → C×1×1"),(.34,.30,"降维 / ReLU / 升维\nC → C/16 → C"),(.74,.24,"Sigmoid 权重\n逐通道相乘")):
        box(ax,x,.09,w,.20,t,size=13)
    arrow(ax,(.245,.19),(.335,.19))
    arrow(ax,(.645,.19),(.735,.19))
    fig.tight_layout()
    save(fig,"components")
    values=np.array([m["per_class_f1"] for m in metrics]).T*100
    idx=np.argsort(values.mean(1))[:10]
    pos=np.arange(10)
    fig,ax=plt.subplots(figsize=(8,4.8))
    ax.barh(pos-.18,values[idx,0],.32,label="ResNet-18",color=COLORS[0])
    ax.barh(pos+.18,values[idx,3],.32,label="DS+SE",color=COLORS[3])
    ax.set_yticks(pos,[labels[i] for i in idx],fontsize=11)
    ax.set(xlabel="测试集 F1 / %",xlim=(0,102))
    ax.invert_yaxis()
    ax.legend(frameon=False,ncol=2,loc="lower center",bbox_to_anchor=(.5,1.01))
    fig.tight_layout()
    save(fig,"class_f1_comparison")
    fig,ax=plt.subplots(figsize=(8,4.3))
    ys=np.arange(len(pairs))
    width=.18
    for k,(name,color) in enumerate(zip(NAMES,COLORS)):
        ax.barh(ys+(k-1.5)*width,[p["counts"][k] for p in pairs],width,label=name,color=color)
    ax.set_yticks(ys,[f'{labels[p["target"]]} → {labels[p["predicted"]]}' for p in pairs],fontsize=10)
    ax.set(xlabel="误分图像数 / 张")
    ax.invert_yaxis()
    ax.legend(ncol=4,frameon=False,loc="lower center",bbox_to_anchor=(.5,1.01))
    fig.tight_layout()
    save(fig,"confusion_pairs")
    # 12 个同图对照分成两组，每组 6 张，正文用可编辑表完整列出数值。
    for group in range(2):
        fig,axes=plt.subplots(2,3,figsize=(7.5,6.1))
        for ax,case in zip(axes.flat,cases[group*6:group*6+6]):
            with Image.open(ROOT/"data/images"/case["path"]) as im:
                ax.imshow(im.convert("RGB"))
            lines=[f'#{case["index"]} 真实：{labels[case["target"]]}']
            for field,name in (("ds","DS"),("combo","DS+SE")):
                r=case[field]
                mark="正确" if r["correct"]=="1" else "误分"
                lines.append(f'{name} {float(r["confidence"])*100:.2f}%（{mark}）')
            ax.set_title("\n".join(lines),fontsize=10.8,pad=8)
            ax.axis("off")
        fig.tight_layout(h_pad=2)
        save(fig,f"scenario_{group+1}")
    # 精度轴完整显示，并另列错误数量；不让截断坐标放大微小收益。
    fig,axes=plt.subplots(1,2,figsize=(8,3.2))
    axes[0].bar(NAMES,[m["accuracy"]*100 for m in metrics],color=COLORS)
    axes[0].set(ylabel="测试准确率 / %",ylim=(0,105))
    for i,m in enumerate(metrics):
        axes[0].text(i,100.8,f'{m["accuracy"]*100:.4f}',ha="center",fontsize=9)
    axes[1].bar(NAMES,[m["errors"] for m in metrics],color=COLORS)
    axes[1].set(ylabel="误分图像数 / 张",ylim=(0,52))
    for i,m in enumerate(metrics):
        axes[1].text(i,m["errors"]+1,str(m["errors"]),ha="center")
    fig.tight_layout()
    save(fig,"accuracy_errors")
    fig,ax=plt.subplots(figsize=(7.1,2.8))
    for tag,label,color in (("0","ε = 0",COLORS[0]),("005","ε = 0.05",COLORS[1])):
        history=rows(ROOT/f"runs/resnet18_smoothing{tag}_s44_15of60/history.csv")
        if len(history)!=15:
            raise RuntimeError("Paired tuning history must have exactly 15 completed epochs")
        ax.plot([int(r["epoch"]) for r in history],[float(r["val_macro_f1"])*100 for r in history],
                "o-",markersize=3,label=label,color=color)
    ax.set(xlabel="epoch",ylabel="验证 Macro-F1 / %",ylim=(0,102),xticks=range(1,16,2))
    ax.legend(frameon=False,ncol=2,loc="lower center",bbox_to_anchor=(.5,1.01))
    ax.text(.99,.06,"训练日志中的 BF16 验证值",transform=ax.transAxes,ha="right",fontsize=10)
    ax.grid(alpha=.18)
    fig.tight_layout()
    save(fig,"tuning_curves")
    print(json.dumps({"subset_pairwise":subset_comparison,"subset_errors":[v["errors"] for v in subset],
                      "scenario_count":len(cases),"validation":validation},ensure_ascii=False))


if __name__ == "__main__":
    main()
