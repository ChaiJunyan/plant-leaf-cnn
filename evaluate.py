"""固定测试协议的评估入口，导出逐图预测、混淆矩阵和分类指标。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import functional as TF

from src.data import LeafDataset, seed_worker
from src.inference import load_model
from src.metrics import confusion_metrics

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    """计算文件指纹，用于核对已冻结的数据清单和最佳权重。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EvaluationDataset(LeafDataset):
    """在固定图像上施加一次预定义扰动，随后使用训练集统计归一化。"""

    def __init__(self, config: dict, condition: str, noise_seed: int):
        norm = config["normalization"]
        super().__init__(ROOT / "data/images", ROOT / "data/manifests/test.csv",
                         config["image_size"], norm["mean"], norm["std"])
        self.condition = condition
        self.noise_seed = noise_seed
        self.config = config

    def __getitem__(self, index):
        if self.condition == "clean":
            return super().__getitem__(index)
        row = self.rows[index]
        with Image.open(self.data_root / row["path"]) as source:
            image = TF.resize(source.convert("RGB"), [self.config["image_size"]] * 2)
        if self.condition == "brightness_0.6":
            image = ImageEnhance.Brightness(image).enhance(0.6)
        elif self.condition == "noise_0.05":
            # 使用清单下标生成独立随机流，各模型读取同一图片时得到同一噪声。
            pixels = np.asarray(image, dtype=np.float32) / 255
            rng = np.random.default_rng(self.noise_seed + index)
            pixels = np.clip(pixels + rng.normal(0, 0.05, pixels.shape), 0, 1)
            image = Image.fromarray(np.rint(pixels * 255).astype(np.uint8))
        elif self.condition == "blur_1.5":
            image = image.filter(ImageFilter.GaussianBlur(radius=1.5))
        else:
            raise ValueError(f"Unsupported condition: {self.condition}")
        norm = self.config["normalization"]
        tensor = TF.normalize(TF.to_tensor(image), norm["mean"], norm["std"])
        return tensor, int(row["label"]), index


def evaluate_run(run_name: str, condition: str, subset: bool, protocol: dict, workers: int):
    """以统一 FP32 推理协议评估一个检查点，并保存可复算的完整输出。"""
    run = ROOT / "runs" / run_name
    if sha256(run / "best.pt") != protocol["checkpoint_sha256"][run_name]:
        raise RuntimeError(f"Frozen checkpoint changed: {run_name}")
    output = ROOT / "results/final" / run_name / ("robustness" if subset else "test") / condition
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    protocol_sha = sha256(ROOT / "configs/final_evaluation_protocol.json")
    if metrics_path.exists():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        if previous["protocol_sha256"] != protocol_sha:
            raise RuntimeError(f"Existing output uses another protocol: {output}")
        for name, expected in previous["output_sha256"].items():
            if not (output / name).exists() or sha256(output / name) != expected:
                raise RuntimeError(f"Saved evaluation output is missing or changed: {output / name}")
        print(f"Already complete: {run_name} {condition} subset={subset}", flush=True)
        return
    torch.set_num_threads(1)
    torch.manual_seed(protocol["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, config = load_model(run / "best.pt", "cuda")
    dataset = EvaluationDataset(config, condition, protocol["noise_seed"])
    selected = protocol["robustness_indices"] if subset else list(range(len(dataset)))
    loader = DataLoader(Subset(dataset, selected), batch_size=protocol["batch_size"],
                        num_workers=workers, pin_memory=False,
                        prefetch_factor=1 if workers else None,
                        worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(protocol["seed"]))
    classes = protocol["classes"]
    matrix = torch.zeros(len(classes), len(classes), dtype=torch.int64)
    rows, probability_batches, ordered_indices = [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for step, (images, targets, indexes) in enumerate(loader, 1):
            probabilities = model(images.cuda()).float().softmax(1).cpu()
            confidence, predicted = probabilities.max(1)
            # 在 CPU 上累计小型整数矩阵，指标计算与模型输出数值精度分离。
            matrix += torch.bincount(targets * len(classes) + predicted,
                                     minlength=len(classes) ** 2).reshape_as(matrix)
            probability_batches.append(probabilities.numpy())
            ordered_indices.extend(indexes.tolist())
            for idx, target, prediction, score in zip(indexes.tolist(), targets.tolist(),
                                                     predicted.tolist(), confidence.tolist()):
                rows.append({"index": idx, "path": dataset.rows[idx]["path"], "target": target,
                             "predicted": prediction, "confidence": score,
                             "correct": int(target == prediction)})
            if step % 20 == 0:
                print(f"{run_name} {condition} batch={step}/{len(loader)}", flush=True)
    if len(rows) != len(selected) or len(set(ordered_indices)) != len(selected):
        raise RuntimeError("Prediction count or sample uniqueness check failed")
    metrics = confusion_metrics(matrix)
    result = {"run": run_name, "condition": condition, "subset": subset, "samples": len(rows),
              "precision": "float32", "cudnn_benchmark": False, "cudnn_deterministic": True,
              "protocol_sha256": protocol_sha, "checkpoint_sha256": protocol["checkpoint_sha256"][run_name],
              "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
              "correct": int(matrix.diagonal().sum()), "errors": len(rows) - int(matrix.diagonal().sum()),
              "per_class_precision": metrics["per_class_precision"],
              "per_class_recall": metrics["per_class_recall"],
              "per_class_f1": metrics["per_class_f1"], "support": metrics["support"],
              "confusion_matrix": matrix.tolist(), "class_names": classes,
              "elapsed_seconds": time.perf_counter() - started, "workers": workers}
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "probabilities.npz", probabilities=np.concatenate(probability_batches),
                        indices=np.asarray(ordered_indices, dtype=np.int64))
    with (output / "per_class.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["class_id", "class_name", "support", "precision", "recall", "f1"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label, name in enumerate(classes):
            writer.writerow({"class_id": label, "class_name": name, "support": int(metrics["support"][label]),
                             "precision": metrics["per_class_precision"][label],
                             "recall": metrics["per_class_recall"][label], "f1": metrics["per_class_f1"][label]})
    result["output_sha256"] = {name: sha256(output / name) for name in
                               ("predictions.csv", "probabilities.npz", "per_class.csv")}
    temporary = metrics_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(metrics_path)
    print(json.dumps({key: result[key] for key in ("run", "condition", "samples", "accuracy", "macro_f1", "errors")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="指定单个训练名称，省略时依次评估协议中的全部模型")
    parser.add_argument("--robustness", action="store_true", help="评估固定分层子集及预定义扰动")
    parser.add_argument("--workers", type=int, default=1, help="加载进程数，内存紧张时可设为0")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "configs/final_evaluation_protocol.json").read_text(encoding="utf-8"))
    if sha256(ROOT / "data/manifests/test.csv") != protocol["test_manifest_sha256"]:
        raise RuntimeError("Test manifest changed after evaluation freeze")
    for name, expected in protocol["evaluation_source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"Frozen evaluation source changed: {name}")
    runs = ([args.run] if args.run else protocol["robustness_runs"] if args.robustness else protocol["runs"])
    if any(run not in protocol["runs"] for run in runs):
        raise ValueError("Requested run is outside the frozen protocol")
    for run in runs:
        conditions = protocol["conditions"] if args.robustness else ["clean"]
        for condition in conditions:
            evaluate_run(run, condition, args.robustness, protocol, args.workers)


if __name__ == "__main__":
    main()
