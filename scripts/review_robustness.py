"""在同一固定子集上复核两个种子的扰动，并扫描模糊半径。"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import functional as TF
from evaluate import EvaluationDataset
from src.data import LeafDataset
from src.inference import load_model
from src.metrics import confusion_metrics

OUTPUT = ROOT / "results/review_20261109"
PROTOCOL = ROOT / "configs/review_20261109/protocol.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


class BlurDataset(EvaluationDataset):
    """沿用原评价的缩放、归一化及清单，只扩展PIL高斯模糊半径。"""

    def __getitem__(self, index):
        if not self.condition.startswith("blur_"):
            return super().__getitem__(index)
        row = self.rows[index]
        with Image.open(self.data_root / row["path"]) as source:
            image = TF.resize(source.convert("RGB"), [self.config["image_size"]]*2)
        image = image.filter(ImageFilter.GaussianBlur(radius=float(self.condition.split("_", 1)[1])))
        norm = self.config["normalization"]
        tensor = TF.normalize(TF.to_tensor(image), norm["mean"], norm["std"])
        return tensor, int(row["label"]), index


def evaluate_case(model, dataset, indices, directory, identity, protocol_hash):
    """保存逐图概率，并从整数混淆矩阵复算指标。"""
    directory.mkdir(parents=True, exist_ok=True)
    saved_path = directory / "metrics.json"
    if saved_path.exists():
        saved = read(saved_path)
        if saved["protocol_sha256"] != protocol_hash:
            raise RuntimeError(f"Existing case uses another protocol: {directory}")
        for filename, digest in saved["output_sha256"].items():
            if sha(directory / filename) != digest:
                raise RuntimeError("Existing prediction output changed")
        return saved
    loader = DataLoader(Subset(dataset, indices), batch_size=128, num_workers=0)
    matrix = torch.zeros(38, 38, dtype=torch.int64)
    rows, arrays, ordered = [], [], []
    begin = time.perf_counter()
    with torch.inference_mode():
        for images, targets, indexes in loader:
            probability = model(images.cuda()).float().softmax(1).cpu()
            confidence, predicted = probability.max(1)
            matrix += torch.bincount(targets*38+predicted, minlength=38**2).reshape(38, 38)
            arrays.append(probability.numpy())
            ordered.extend(indexes.tolist())
            for index, target, prediction, score in zip(indexes.tolist(), targets.tolist(),
                                                       predicted.tolist(), confidence.tolist()):
                rows.append({"index": index, "path": dataset.rows[index]["path"], "target": target,
                             "predicted": prediction, "confidence": score, "correct": int(target == prediction)})
    if ordered != indices:
        raise RuntimeError("Evaluation order differs from frozen indices")
    independent = np.zeros((38, 38), dtype=np.int64)
    for row in rows:
        independent[row["target"], row["predicted"]] += 1
    assert np.array_equal(independent, matrix.numpy())
    metrics = confusion_metrics(matrix)
    with (directory / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(directory / "probabilities.npz", probabilities=np.concatenate(arrays), indices=np.array(ordered))
    result = {**identity, "samples": len(rows), "precision": "float32", "protocol_sha256": protocol_hash,
              **metrics, "confusion_matrix": matrix.tolist(), "errors": int(len(rows)-matrix.diagonal().sum()),
              "elapsed_seconds": time.perf_counter()-begin,
              "output_sha256": {name: sha(directory/name) for name in ("predictions.csv", "probabilities.npz")}}
    write(saved_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args()
    protocol = read(PROTOCOL)
    for filename, digest in protocol["source_sha256"].items():
        if sha(ROOT/filename) != digest:
            raise RuntimeError(f"Frozen study source changed: {filename}")
    if sha(ROOT/"data/manifests/test.csv") != protocol["test_manifest_sha256"]:
        raise RuntimeError("Test manifest changed")
    torch.set_num_threads(1)
    torch.manual_seed(20261109)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    records, calibration = [], []
    progress_path = OUTPUT/"robustness_progress.json"
    protocol_hash = sha(PROTOCOL)
    for run in protocol["runs"]:
        checkpoint = ROOT/"runs"/run/"best.pt"
        if sha(checkpoint) != protocol["checkpoint_sha256"][run]:
            raise RuntimeError(f"Checkpoint changed: {run}")
        model = config = None
        for condition in protocol["conditions"]:
            if time.time() >= args.deadline_unix or (ROOT/"STOP_TRAINING").exists():
                write(progress_path, {"status": "budget_reached", "completed_cases": len(records), "run": run})
                return
            write(progress_path, {"status": "running", "run": run, "condition": condition,
                                  "completed_cases": len(records), "total_cases": len(protocol["runs"])*len(protocol["conditions"]),
                                  "updated_utc": datetime.now(timezone.utc).isoformat()})
            cached = ROOT/"results/final"/run/"robustness"/condition/"metrics.json"
            if cached.exists():
                saved = read(cached)
                if saved["checkpoint_sha256"] != protocol["checkpoint_sha256"][run] or saved["samples"] != 1140:
                    raise RuntimeError("Original robustness cache differs")
                arrays = np.load(cached.parent/"probabilities.npz")
                assert arrays["indices"].tolist() == protocol["robustness_indices"]
                for filename, digest in saved["output_sha256"].items():
                    assert sha(cached.parent/filename) == digest
                result_path = cached
            else:
                if model is None:
                    model, config = load_model(checkpoint, "cuda")
                dataset = BlurDataset(config, condition, protocol["noise_seed"])
                directory = OUTPUT/run/"robustness"/condition
                evaluate_case(model, dataset, protocol["robustness_indices"], directory,
                              {"run": run, "condition": condition, "subset": True,
                               "checkpoint_sha256": protocol["checkpoint_sha256"][run]}, protocol_hash)
                result_path = directory/"metrics.json"
            record = read(result_path)
            records.append({"run": run, "condition": condition, "metrics_path": str(result_path.relative_to(ROOT)),
                            "accuracy": record["accuracy"], "macro_f1": record["macro_f1"], "errors": record["errors"],
                            "metrics_sha256": sha(result_path)})
            print(json.dumps(records[-1]), flush=True)
        if run in protocol["calibration_runs"]:
            if model is None:
                model, config = load_model(checkpoint, "cuda")
            if sha(ROOT/"data/manifests/val.csv") != protocol["validation_manifest_sha256"]:
                raise RuntimeError("Validation manifest changed")
            norm = config["normalization"]
            dataset = LeafDataset(ROOT/"data/images", ROOT/"data/manifests/val.csv", 224, norm["mean"], norm["std"])
            directory = OUTPUT/run/"calibration"
            evaluated = evaluate_case(model, dataset, list(range(len(dataset))), directory,
                                      {"run": run, "condition": "validation_clean", "subset": False,
                                       "checkpoint_sha256": protocol["checkpoint_sha256"][run]}, protocol_hash)
            array = np.load(directory/"probabilities.npz")["probabilities"]
            scores = array.max(1)
            threshold = float(np.quantile(scores, protocol["rejection_quantile"], method="lower"))
            calibration.append({"run": run, "threshold": threshold, "quantile": protocol["rejection_quantile"],
                                "known_validation_coverage": float(np.mean(scores >= threshold)),
                                "metrics_path": str((directory/"metrics.json").relative_to(ROOT)),
                                "samples": evaluated["samples"]})
        del model
        torch.cuda.empty_cache()
    write(OUTPUT/"robustness_summary.json", {"complete": True, "cases": records, "calibration": calibration,
                                             "protocol_sha256": protocol_hash})
    write(progress_path, {"status": "complete", "completed_cases": len(records), "calibration_models": len(calibration)})


if __name__ == "__main__":
    main()
