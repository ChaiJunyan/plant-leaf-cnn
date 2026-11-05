"""在统一硬件、精度与输入尺寸下测量模型推理耗时，保留逐次原始读数。"""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.inference import load_model


def summarize(values):
    """返回实测延迟的分位数，单位为毫秒。"""
    return {"median_ms": float(np.median(values)), "p05_ms": float(np.percentile(values, 5)),
            "p95_ms": float(np.percentile(values, 95)), "mean_ms": float(np.mean(values)),
            "std_ms": float(np.std(values, ddof=1)), "samples": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=50, help="每一轮的测量次数")
    parser.add_argument("--rounds", type=int, default=3, help="轮换模型顺序后的独立测量轮数")
    args = parser.parse_args()
    if args.repeats < 2 or args.rounds < 1:
        raise ValueError("At least two repetitions are required")
    protocol = json.loads((ROOT / "configs/final_evaluation_protocol.json").read_text(encoding="utf-8"))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    result = {"precision": "float32", "tf32": False, "image_size": 224, "warmup": 20,
              "repeats_per_round": args.repeats, "rounds": args.rounds,
              "cpu_threads": 4, "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "platform": platform.platform(),
              "scope": "Model-forward only. GPU inputs are already on device. Image decoding, resizing, transfer and Grad-CAM are excluded. GPU event time and synchronized wall time are both retained.",
              "runs": {}}
    runs = protocol["robustness_runs"]
    execution_order = []
    for round_index in range(args.rounds):
        offset = round_index % len(runs)
        execution_order.extend(runs[offset:] + runs[:offset])
    for run in execution_order:
        model, _ = load_model(ROOT / "runs" / run / "best.pt", "cuda")
        record = {}
        with torch.inference_mode():
            for batch in (1, 32):
                torch.manual_seed(protocol["seed"] + batch)
                x = torch.randn(batch, 3, 224, 224, device="cuda")
                for _ in range(result["warmup"]):
                    model(x)
                torch.cuda.synchronize()
                gpu_times, wall_times = [], []
                for _ in range(args.repeats):
                    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    wall_start = time.perf_counter()
                    start.record()
                    model(x)
                    end.record()
                    end.synchronize()
                    wall_times.append((time.perf_counter() - wall_start) * 1000)
                    gpu_times.append(start.elapsed_time(end))
                record[f"gpu_batch{batch}"] = {"cuda_event": summarize(gpu_times),
                                               "synchronized_wall": summarize(wall_times)}
            model.cpu()
            torch.manual_seed(protocol["seed"] + 1)
            x = torch.randn(1, 3, 224, 224)
            for _ in range(result["warmup"]):
                model(x)
            cpu_times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                model(x)
                cpu_times.append((time.perf_counter() - start) * 1000)
            record["cpu_batch1"] = summarize(cpu_times)
        round_medians = {"cpu_ms": record["cpu_batch1"]["median_ms"],
                         "gpu_ms": record["gpu_batch1"]["synchronized_wall"]["median_ms"]}
        previous = result["runs"].get(run)
        if previous is not None:
            for case in ("gpu_batch1", "gpu_batch32"):
                for scope in ("cuda_event", "synchronized_wall"):
                    record[case][scope] = summarize(previous[case][scope]["samples"] + record[case][scope]["samples"])
            record["cpu_batch1"] = summarize(previous["cpu_batch1"]["samples"] + record["cpu_batch1"]["samples"])
        record["round_medians"] = (previous["round_medians"] if previous else []) + [round_medians]
        result["runs"][run] = record
        result["complete"] = all(len(result["runs"].get(name, {}).get("round_medians", [])) == args.rounds for name in runs)
        output = ROOT / "results/final/latency.json"
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"run": run, "gpu_batch1_ms": record["gpu_batch1"]["synchronized_wall"]["median_ms"],
                          "cpu_batch1_ms": record["cpu_batch1"]["median_ms"]}), flush=True)


if __name__ == "__main__":
    main()
