# 完整复现步骤

以下命令在项目根目录执行。训练使用 Python 3.10、PyTorch 2.6.0、torchvision 0.21.0，以及支持 BF16 的 CUDA 显卡。数据与检查点建议预留至少15GB空间。

## 环境和数据

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/freeze_splits.py
python scripts/verify_models.py
```

数据准备会生成 `data/manifests/`。冻结划分后，训练、验证、测试和隔离集分别为37,798、5,498、10,709和300张。

## 六组主实验与评价

```powershell
python -u scripts/run_experiments.py --protocol configs/reproduction_protocol.json --max-hours 8
```

队列包含种子42的四种结构，以及基线和 DS+SE 的种子43重复，每组60轮。运行预算到期后在完整轮次结束时保存，再次执行相同命令可继续。完成全部六组后执行以下步骤。

```powershell
$cnnRuns = 'resnet18_s42_224','resnet18_ds_s42_224','resnet18_se_s42_224','resnet18_ds_se_s42_224','resnet18_s43_224','resnet18_ds_se_s43_224'
foreach ($cnnRun in $cnnRuns) { python scripts/verify_checkpoint.py --run $cnnRun }
python scripts/freeze_evaluation.py
python evaluate.py
python evaluate.py --robustness
python scripts/measure_latency.py
python scripts/summarize_results.py
python scripts/verify_results.py
```

汇总脚本生成 `results/final/summary.json`。已有历史汇总会核对后保留原字节，维持冻结协议的指纹一致性。时延按当前硬件重测，数值会受设备与系统负载影响。

## 补充重复与标签平滑

```powershell
python -u train.py --config configs/supplement_20261106/resnet18_ds_s43_224.json
python -u train.py --config configs/supplement_20261106/resnet18_se_s43_224.json
python -u train.py --config configs/supplement_20261106/smoothing0_s44_15of60.json --stop-after-epochs 15
python -u train.py --config configs/supplement_20261106/smoothing005_s44_15of60.json --stop-after-epochs 15
python scripts/verify_checkpoint.py --run resnet18_ds_s43_224
python scripts/verify_checkpoint.py --run resnet18_se_s43_224
python scripts/evaluate_additional.py --run resnet18_ds_s43_224 --run resnet18_se_s43_224
python scripts/evaluate_tuning_validation.py
python scripts/summarize_results.py
python scripts/export_inference_weights.py --overwrite
```

第二次汇总生成 `results/supplement_20261106/summary.json`，覆盖四种结构的双种子结果，补齐报告表5.4的生成步骤。两组标签平滑实验均执行60轮计划的前15轮，只评价验证集，配对数值采用 FP32 重载结果。导出时用 `--overwrite` 将仓库内演示权重替换为本次重训权重，也可用 `--output <目录>` 单独导出。

## 双种子扰动、学习率与外部案例

```powershell
python scripts/freeze_review_protocol.py
python -u scripts/run_review_study.py --max-hours 10
python scripts/download_external_review.py
python scripts/review_external.py
python scripts/verify_evidence.py --directory .
```

队列完成72组固定子集评价，并新增学习率0.02、0.1两组15轮验证对照。0.05复用此前的标签平滑组。预算到期后再次执行队列可续跑。外部图片按固定来源下载，阈值取完整验证集最高 softmax 分数的第5百分位。

## 中断与重新评价

单次训练可用 `python train.py --config <配置路径> --resume runs/<运行名>/last.pt` 恢复。提前停止可在根目录创建 `STOP_TRAINING`，续跑前移走该文件。强制中断后先确认旧进程已退出，再处理遗留的 `runs/queue.lock`。

重新建立主评价协议时执行 `python scripts/begin_new_evaluation.py`，它会归档已有协议和结果。固定种子与环境用于控制实验设置，GPU算法和数据加载恢复仍可能产生数值差异。原始报告结果可始终通过仓库内的复核材料重算。

## 报告补充分析

```powershell
python scripts/analyze_posthoc.py
python scripts/analyze_posthoc.py --profile-stages
```

第一条只读取原始复核包，用已保存的逐图输出复算准确率配对、按关联组bootstrap区间和15箱ECE。第二条额外用PyTorch统计四种替换位置的参数与MACs，新增位置的统计对应未训练结构。补充分析结果位于 `evidence/posthoc_20261110/`。
