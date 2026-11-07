"""等待补充训练结束，复核检查点并在独立目录评价完整重复。

短预算调参只汇总验证结果；原始六组结果和报告不会被此脚本覆盖。
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "results/supplement_20261106"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def history(name):
    path = ROOT / "runs" / name / "history.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def evaluate(name, protocol, protocol_path):
    """独立保存新检查点的FP32测试结果，并从预测重建矩阵核对。"""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from src.data import LeafDataset, seed_worker
    from src.inference import load_model
    from src.metrics import confusion_metrics

    destination = OUTPUT / name / "test/clean"
    destination.mkdir(parents=True, exist_ok=True)
    metric_path = destination / "metrics.json"
    protocol_hash = sha(protocol_path)
    if metric_path.exists():
        old = read(metric_path)
        if old["protocol_sha256"] != protocol_hash:
            raise RuntimeError("Saved supplemental result has a different protocol")
        for filename, digest in old["output_sha256"].items():
            if sha(destination/filename) != digest:
                raise RuntimeError("Saved supplemental output changed")
        return old
    checkpoint = ROOT / "runs" / name / "best.pt"
    if sha(checkpoint) != protocol["checkpoint_sha256"][name]:
        raise RuntimeError("Checkpoint changed after supplemental evaluation freeze")
    torch.set_num_threads(1)
    torch.manual_seed(protocol["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, config = load_model(checkpoint, "cuda")
    norm = config["normalization"]
    data = LeafDataset(ROOT/"data/images", ROOT/"data/manifests/test.csv",
                       config["image_size"],norm["mean"],norm["std"])
    loader = DataLoader(data,batch_size=protocol["batch_size"],num_workers=1,
                        pin_memory=False,prefetch_factor=1,worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(protocol["seed"]))
    matrix = torch.zeros(38,38,dtype=torch.int64)
    records, arrays, indexes_all = [], [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for step,(images,targets,indexes) in enumerate(loader,1):
            probabilities=model(images.cuda()).float().softmax(1).cpu()
            confidence,predicted=probabilities.max(1)
            matrix += torch.bincount(targets*38+predicted,minlength=38**2).reshape(38,38)
            arrays.append(probabilities.numpy())
            indexes_all.extend(indexes.tolist())
            for i,t,p,c in zip(indexes.tolist(),targets.tolist(),predicted.tolist(),confidence.tolist()):
                records.append({"index":i,"path":data.rows[i]["path"],"target":t,
                                "predicted":p,"confidence":c,"correct":int(t==p)})
            if step%20==0: print(f"Supplemental {name} batch={step}/{len(loader)}",flush=True)
    assert indexes_all==list(range(len(data)))
    recomputed=np.zeros((38,38),dtype=np.int64)
    for r in records: recomputed[r["target"],r["predicted"]]+=1
    assert np.array_equal(recomputed,matrix.numpy())
    probabilities=np.concatenate(arrays)
    assert np.array_equal(probabilities.argmax(1),[r["predicted"] for r in records])
    m=confusion_metrics(matrix)
    independent_f1=float((2*recomputed.diagonal()/np.maximum(recomputed.sum(0)+recomputed.sum(1),1)).mean())
    assert abs(independent_f1-m["macro_f1"])<1e-12
    with (destination/"predictions.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    np.savez_compressed(destination/"probabilities.npz",probabilities=probabilities,indices=np.array(indexes_all))
    with (destination/"per_class.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["class_id","class_name","support","precision","recall","f1"])
        writer.writeheader()
        for k,label in enumerate(protocol["classes"]):
            writer.writerow({"class_id":k,"class_name":label,"support":int(m["support"][k]),
                             "precision":m["per_class_precision"][k],"recall":m["per_class_recall"][k],
                             "f1":m["per_class_f1"][k]})
    result={"run":name,"condition":"clean","subset":False,"samples":len(data),"precision":"float32",
            "protocol_sha256":protocol_hash,"checkpoint_sha256":protocol["checkpoint_sha256"][name],
            **m,"correct":int(matrix.diagonal().sum()),"errors":len(data)-int(matrix.diagonal().sum()),
            "confusion_matrix":matrix.tolist(),"class_names":protocol["classes"],
            "elapsed_seconds":time.perf_counter()-start,"independent_prediction_audit_passed":True,
            "output_sha256":{filename:sha(destination/filename) for filename in
                             ("predictions.csv","probabilities.npz","per_class.csv")}}
    write(metric_path,result)
    print(json.dumps({k:result[k] for k in ("run","accuracy","macro_f1","errors")}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait",action="store_true",help="后台等待训练队列结束；等待时不导入torch、不占用GPU")
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    status_path=OUTPUT/"postprocess_status.json"
    status={"status":"waiting_for_training","pid":__import__("os").getpid(),"started_utc":datetime.now(timezone.utc).isoformat()}
    write(status_path,status)
    sleep_request=None
    try:
        if sys.platform=="win32":
            import ctypes
            sleep_request=ctypes.windll.kernel32.SetThreadExecutionState
            sleep_request.argtypes=[ctypes.c_ulong]
            sleep_request.restype=ctypes.c_ulong
            if not sleep_request(0x80000001):raise OSError("Unable to inhibit idle sleep during postprocessing")
        queue=ROOT/"runs/supplement_status_20261106.json"
        while True:
            current=read(queue)
            if current["status"]!="running":break
            if not args.wait:
                raise RuntimeError("Training is active; use --wait or run after completion")
            time.sleep(30)
        if current["status"]=="failed":
            raise RuntimeError(f"Supplemental training failed: {current.get('error')}")
        if (ROOT/"runs/queue.lock").exists():
            raise RuntimeError("Another training queue holds the GPU lock; postpone evaluation")
        supplement=read(ROOT/"configs/supplement_20261106/protocol.json")
        original=read(ROOT/"configs/final_evaluation_protocol.json")
        for name,digest in supplement["source_sha256"].items():
            if sha(ROOT/name)!=digest:raise RuntimeError(f"Frozen source changed: {name}")
        for name,digest in supplement["manifest_sha256"].items():
            if sha(ROOT/name)!=digest:raise RuntimeError(f"Frozen manifest changed: {name}")
        complete,skipped=[],[]
        for task in supplement["tasks"]:
            config=read(ROOT/task["config"]);name=config["run_name"];rows=history(name)
            if [int(r["epoch"]) for r in rows]!=list(range(1,len(rows)+1)):
                raise RuntimeError(f"Nonconsecutive history: {name}")
            if len(rows)<task["target_epochs"]:
                skipped.append({"run":name,"completed_epochs":len(rows),"target_epochs":task["target_epochs"]})
                continue
            complete.append((task,config,rows))
        status.update(status="verifying_checkpoints",complete_runs=[c[1]["run_name"] for c in complete],skipped=skipped)
        write(status_path,status)
        for task,config,rows in complete:
            name=config["run_name"]
            subprocess.run([sys.executable,"-u","scripts/verify_checkpoint.py","--run",name],cwd=ROOT,check=True)
            if not read(ROOT/"runs"/name/"checkpoint_validation_check.json")["matches_saved_metrics"]:
                raise RuntimeError(f"Checkpoint validation mismatch: {name}")
        repeats=[config["run_name"] for task,config,rows in complete if task["kind"]=="second_seed_repeat"]
        path=ROOT/"configs/supplement_20261106/evaluation_protocol.json"
        frozen={"runs":repeats,"seed":original["seed"],"batch_size":original["batch_size"],
                "classes":original["classes"],"precision":"float32","cudnn_benchmark":False,
                "cudnn_deterministic":True,"test_manifest_sha256":sha(ROOT/"data/manifests/test.csv"),
                "checkpoint_sha256":{name:sha(ROOT/"runs"/name/"best.pt") for name in repeats},
                "source_sha256":{name:sha(ROOT/name) for name in
                                 ("scripts/finish_supplemental.py","src/data.py","src/models.py","src/inference.py","src/metrics.py")},
                "training_protocol_sha256":sha(ROOT/"configs/supplement_20261106/protocol.json"),
                "post_hoc_supplement":True,"pilots_are_validation_only":True}
        if path.exists():
            protocol=read(path)
            if any(protocol.get(k)!=v for k,v in frozen.items()):
                raise RuntimeError("Supplemental evaluation protocol already exists with different inputs")
        else:
            protocol={**frozen,"frozen_utc":datetime.now(timezone.utc).isoformat()};write(path,protocol)
        status["status"]="evaluating_complete_repeats";write(status_path,status)
        evaluations=[evaluate(name,protocol,path) for name in repeats]
        originals=read(ROOT/"results/final/summary.json")["all_runs"]
        all_runs=[{k:r[k] for k in ("run","model","seed","accuracy","macro_f1","errors")} for r in originals]
        for r in evaluations:
            all_runs.append({"run":r["run"],"model":r["run"].rsplit("_s",1)[0],"seed":43,
                             "accuracy":r["accuracy"],"macro_f1":r["macro_f1"],"errors":r["errors"]})
        aggregate=[]
        for model in ("resnet18","resnet18_ds","resnet18_se","resnet18_ds_se"):
            group=[r for r in all_runs if r["model"]==model]
            aggregate.append({"model":model,"n":len(group),
                              **{f"{metric}_{stat}":function([r[metric] for r in group])
                                 for metric in ("accuracy","macro_f1")
                                 for stat,function in (("mean",statistics.mean),("std",statistics.stdev))
                                 if len(group)>1}})
        pilots=[]
        for task,config,rows in complete:
            if task["kind"]!="paired_validation_only_tuning":continue
            v=read(ROOT/"runs"/config["run_name"]/"best_validation.json")
            pilots.append({"run":config["run_name"],"label_smoothing":config["label_smoothing"],
                           "completed_epochs":len(rows),"lr_horizon":config["epochs"],"best_epoch":v["epoch"],
                           "best_val_accuracy":v["accuracy"],"best_val_macro_f1":v["macro_f1"],
                           "last_val_accuracy":float(rows[-1]["val_accuracy"]),
                           "last_val_macro_f1":float(rows[-1]["val_macro_f1"]),"test_evaluated":False})
        write(OUTPUT/"summary.json",{"all_runs":all_runs,"repeats":aggregate,"validation_pilots":pilots,
                                      "paired_pilots_complete":len(pilots)==2,"skipped":skipped,
                                      "post_hoc_supplement":True,"protocol_sha256":sha(path)})
        if sha(ROOT/"results/final/summary.json")!=supplement["original_summary_sha256"]:
            raise RuntimeError("Original summary changed during supplemental work")
        status.update(status="complete" if not skipped else "partial_budget_reached",finished_utc=datetime.now(timezone.utc).isoformat(),
                      summary="results/supplement_20261106/summary.json")
        write(status_path,status)
        print(json.dumps(status,ensure_ascii=False),flush=True)
    except BaseException as error:
        status.update(status="failed",error=repr(error));write(status_path,status);raise
    finally:
        if sleep_request is not None:sleep_request(0x80000000)


if __name__=="__main__":
    main()
