# ResNet-18 叶片病害识别与结构改进

神经网络与深度学习课程大作业代码。以经典 ResNet-18 为基础，比较深度可分离卷积（DS）、SE 通道注意力及二者组合；场景应用采用 Python 终端单图预测。

仓库提供源代码、实验配置和运行说明。报告、结果图、原始数据、训练记录及模型权重保存在本机，不随仓库上传。实验结果和分析见课程报告。

## 环境

本次实际环境：Windows、Python3.10、RTX4060 Laptop 8GB、PyTorch2.6.0+cu124、torchvision0.21.0+cu124。创建Python3.10环境后，在项目根目录安装依赖：

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

单图预测默认使用CPU；完整训练与评价按CUDA实现。图表使用宋体和Times New Roman，其他系统需配置可用中文字体。

## 数据准备

```powershell
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/freeze_splits.py
```

数据来自 [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage)，下载版本由 `configs/dataset_source.json` 固定为 `9e97599868962bd0079b8db4b7f1efa9185fa1e7`，支持断点续传和SHA256校验。原始压缩包约2.18GB（十进制）。建议预留至少15GB用于下载、解码原图及训练检查点，这是运行空间建议。

准备脚本只解码color图像，结合叶片元数据、相同像素与dHash候选生成分组划分，保守隔离关联样本，并仅从最终训练集统计归一化参数。输出位于 `data/manifests/`，训练配置保留清单哈希，若重新生成的数据与配置不一致会停止。不要用新的随机划分直接引用原报告结果。

上游声明 [CC-BY-SA-3.0](https://creativecommons.org/licenses/by-sa/3.0/)。本仓库不重新分发原图。引用：Mohanty S P, Hughes D P, Salathé M. *Using Deep Learning for Image-Based Plant Disease Detection*. Frontiers in Plant Science, 2016, 7:1419. https://doi.org/10.3389/fpls.2016.01419

## 训练与断点恢复

```powershell
python scripts/verify_models.py
python -u scripts/run_experiments.py --protocol configs/reproduction_protocol.json --max-hours 8
```

复现实验配置对应实际完成的六次训练：种子42四组主实验，以及基线和DS+SE的种子43重复，各60轮。原实验中DS+SE由验证集选定用于重点重复；复现协议固定同样六组，便于逐项比较。模型均随机初始化，输入224×224、batch96、SGD、学习率0.05、标签平滑0.05、预热3轮后余弦衰减。DS修改layer3/layer4残差分支的3×3卷积，SE的reduction为16。

8小时是当前会话预算，达到后在完整轮次结束时保存，不代表8小时能完成六次训练。再次运行相同命令会恢复尚未完成的实验。完成状态为 `runs/queue_status.json` 中的 `all_training_complete`。

```powershell
Get-Content runs/queue_status.json
# 替换为当前run名称
Get-Content runs/resnet18_s42_224/console_formal.log -Tail 12
Get-Content runs/resnet18_s42_224/history.csv -Tail 3
```

单独训练或续跑：

```powershell
python -u train.py --config configs/formal/resnet18_s42_224.json
python -u train.py --config configs/formal/resnet18_s42_224.json --resume runs/resnet18_s42_224/last.pt
```

提前停止可在根目录创建空文件 `STOP_TRAINING`，当前完整轮次结束后保存退出。续跑前移走该文件。强制中断后先确认旧训练进程退出，再处理遗留 `runs/queue.lock`，避免启动重复队列。

## 检查点复核与最终评价

最佳检查点按验证Macro-F1选择，平分时比较准确率，再保留较早轮次。训练队列结束后，先重载六组最佳权重复核验证指标：

```powershell
$cnnRuns = 'resnet18_s42_224','resnet18_ds_s42_224','resnet18_se_s42_224','resnet18_ds_se_s42_224','resnet18_s43_224','resnet18_ds_se_s43_224'
foreach ($cnnRun in $cnnRuns) { python scripts/verify_checkpoint.py --run $cnnRun }
python scripts/freeze_evaluation.py
python evaluate.py
python evaluate.py --robustness
python scripts/measure_latency.py
python scripts/make_final_artifacts.py
python scripts/make_explanations.py
python scripts/verify_results.py
```

冻结脚本在首次测试前记录检查点、源文件和测试清单的哈希，以及固定分层扰动子集。最终测试统一FP32、关闭TF32，输出逐图CSV、概率NPZ、逐类CSV和指标JSON。`verify_results.py` 从保存的预测独立复算并交叉核对，不需要GPU。MACs只统计Conv2d/Linear，1MAC=2次算术FLOPs，实际耗时另测。

若已有旧评估而要评价一批新训练权重，使用 `python scripts/begin_new_evaluation.py` 替代直接冻结；它先将旧协议和结果归档到带时间戳的 `records/evaluation_*`。不要修改旧协议中的哈希来绕过检查。

固定种子不承诺跨设备逐位相同。原训练使用BF16、cuDNN benchmark，且数据加载进程数因Windows内存约束做过记录化调整。报告保留原始运行的真实数字；重新训练产生的新数字应单独记录。

## 单图场景应用

已有本机导出的 `weights/leaf_ds_se.pt` 和样例时：

```powershell
python predict.py --image assets/examples/example_0.jpg
# 自己的图片
python predict.py --image "D:\photos\leaf.jpg"
# 使用其他训练权重
python predict.py --image "D:\photos\leaf.jpg" --checkpoint runs/resnet18_ds_se_s42_224/best.pt
```

输出为终端前五类分数、`results/demo/prediction.json` 和 `prediction.png`。纯代码仓库不附权重和样例图；新环境可按上述流程重训，再由 `make_final_artifacts.py` 导出权重与验证样例，或在 `--checkpoint` 中指定已有权重。结果图可直接用于课堂展示，无需网页服务。softmax分数用于展示分类器输出，真实田间适用性需独立数据验证。

## 文件导航

| 文件 | 功能 |
|---|---|
| `src/models.py` | ResNet-18、DW+PW、SE及参数/MACs统计 |
| `src/data.py` | 数据读取、增强和归一化 |
| `train.py` | 训练、验证、学习率与断点恢复 |
| `evaluate.py` | 固定FP32测试和输入扰动 |
| `predict.py` | 终端单图推理与结果图片 |
| `scripts/verify_models.py` | 基线等价性、梯度和实际参数更新检查 |
| `scripts/verify_checkpoint.py` | 最佳权重与验证日志复核 |
| `scripts/verify_results.py` | 预测、混淆矩阵和指标独立复算 |
| `scripts/measure_latency.py` | 轮换顺序的CPU/GPU前向计时 |
| `scripts/make_final_artifacts.py` | 真实输出汇总、绘图和权重导出 |
| `scripts/make_explanations.py` | 固定规则选择样本的Grad-CAM |

结构参考：ResNet（He等，CVPR2016）、MobileNets（Howard等，2017）、SE（Hu等，CVPR2018）、Grad-CAM（Selvaraju等，ICCV2017）。
