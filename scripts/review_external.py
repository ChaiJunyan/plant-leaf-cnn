"""使用公开推理权重展示外部图像top5及验证集确定的拒识规则。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from src.inference import load_model, prepare_image

OUTPUT = ROOT/"results/review_20261109"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = read(ROOT/"assets/external_review_sources.json")
    if not manifest["download_complete"]:
        raise RuntimeError("External image download is incomplete")
    calibration = read(OUTPUT/"robustness_summary.json")["calibration"]
    thresholds = {row["run"]: row["threshold"] for row in calibration}
    weights = read(ROOT/"weights/inference_manifest.json")["models"]
    frozen = {"image_manifest_sha256": sha(ROOT/"assets/external_review_sources.json"),
              "calibration_summary_sha256": sha(OUTPUT/"robustness_summary.json"),
              "weights": weights, "thresholds": thresholds,
              "source_sha256": {name: sha(ROOT/name) for name in
                                ("scripts/review_external.py", "src/inference.py", "src/models.py")},
              "preprocessing": "Original RGB decode, resize224x224, training normalization; identical to existing CLI.",
              "precision": "float32", "device": "cpu", "no_external_label_threshold_fitting": True}
    protocol_path = OUTPUT/"external_protocol.json"
    if protocol_path.exists():
        protocol = read(protocol_path)
        if any(protocol.get(key) != value for key, value in frozen.items()):
            raise RuntimeError("Frozen external protocol inputs changed")
    else:
        protocol = {**frozen, "frozen_utc": datetime.now(timezone.utc).isoformat()}
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    torch.set_num_threads(2)
    labels = read(ROOT/"assets/labels_zh.json")
    cases = [{**record, "models": {}} for record in manifest["images"]]
    probabilities = []
    order = []
    for exported in weights:
        weight_path = ROOT/"weights"/exported["filename"]
        if sha(weight_path) != exported["sha256"]:
            raise RuntimeError("Public inference weight changed")
        model, config = load_model(weight_path, "cpu")
        for case in cases:
            image_path = ROOT/"data/external_review_20261109"/case["filename"]
            if sha(image_path) != case["sha256"]:
                raise RuntimeError("External image changed")
            with Image.open(image_path) as source:
                tensor = prepare_image(source.convert("RGB"), config).unsqueeze(0)
            with torch.inference_mode():
                first = model(tensor)
                second = model(tensor)
                if not torch.equal(first, second):
                    raise RuntimeError("Consecutive CPU predictions differ")
                probability = first.softmax(1)[0].numpy()
            ranking = np.argsort(-probability)[:5]
            threshold = thresholds[exported["run"]]
            score = float(probability[ranking[0]])
            prediction = {"top5": [{"label": int(index), "name": labels[index], "score": float(probability[index])}
                                    for index in ranking], "max_softmax": score, "threshold": threshold,
                          "accepted": score >= threshold,
                          "top1_top2_margin": float(probability[ranking[0]]-probability[ranking[1]]),
                          "entropy_nats": float(-np.sum(probability*np.log(np.clip(probability, 1e-12, 1))))}
            case["models"][exported["model"]] = prediction
            probabilities.append(probability)
            order.append({"image": case["id"], "model": exported["model"]})
            print(json.dumps({"id": case["id"], "model": exported["model"], "top1": prediction["top5"][0],
                              "accepted": prediction["accepted"]}, ensure_ascii=False), flush=True)
        del model
    np.savez_compressed(OUTPUT/"external_probabilities.npz", probabilities=np.array(probabilities))
    result = {"complete": True, "cases": cases, "probability_order": order,
              "protocol_sha256": sha(protocol_path), "calibration": calibration,
              "probabilities_sha256": sha(OUTPUT/"external_probabilities.npz"),
              "labels_are_source_descriptions": True, "disease_accuracy_estimated": False}
    (OUTPUT/"external_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
