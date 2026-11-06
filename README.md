# ResNet-18 叶片病害识别

基于 PlantVillage 数据集实现叶片图像分类，支持标准 ResNet-18、深度可分离卷积（DS）、SE 通道注意力及 DS+SE 四种结构，提供训练、评估、性能测量和命令行单图预测功能。

## 环境安装

使用 Python 3.10，在项目根目录执行：

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

训练和批量评估使用 CUDA，单图预测默认使用 CPU。绘图使用宋体和 Times New Roman，其他系统可在绘图脚本中调整字体。

## 数据准备

```powershell
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/freeze_splits.py
```

下载脚本根据 `configs/dataset_source.json` 获取固定版本的数据，支持断点续传和 SHA256 校验。数据准备脚本完成图像解码、叶片分组、重复样本检查及训练集归一化统计，划分清单保存到 `data/manifests/`。建议预留至少 15GB 磁盘空间用于数据和训练检查点。

数据来源：[mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage)，固定版本为 `9e97599868962bd0079b8db4b7f1efa9185fa1e7`，上游许可为 [CC-BY-SA-3.0](https://creativecommons.org/licenses/by-sa/3.0/)。数据引用：Mohanty S P, Hughes D P, Salathé M. *Using Deep Learning for Image-Based Plant Disease Detection*. Frontiers in Plant Science, 2016, 7:1419. [DOI](https://doi.org/10.3389/fpls.2016.01419)。

## 模型与训练

| 配置名称 | 结构 |
|---|---|
| `resnet18` | 标准 ResNet-18 |
| `resnet18_ds` | 将 layer3、layer4 残差分支的 3×3 卷积替换为 DW+PW |
| `resnet18_se` | 在残差分支末端 BN 之后、残差相加之前加入 SE |
| `resnet18_ds_se` | 同时使用 DS 与 SE |

检查模型实现并启动训练队列：

```powershell
python scripts/verify_models.py
python -u scripts/run_experiments.py --protocol configs/reproduction_protocol.json --max-hours 8
```

队列包含种子 42 的四组模型，以及基线和 DS+SE 的种子 43 重复训练，每组 60 轮。`--max-hours 8` 限制本次运行时长，到期后在完整轮次结束时保存；再次执行相同命令会继续训练。

查看进度：

```powershell
Get-Content runs/queue_status.json
# 替换为当前正在训练的 run 名称
Get-Content runs/resnet18_s42_224/console_formal.log -Tail 12
Get-Content runs/resnet18_s42_224/history.csv -Tail 3
```

单独训练或从检查点恢复：

```powershell
python -u train.py --config configs/formal/resnet18_s42_224.json
python -u train.py --config configs/formal/resnet18_s42_224.json --resume runs/resnet18_s42_224/last.pt
```

需要提前停止时，在项目根目录创建空文件 `STOP_TRAINING`，程序将在完整轮次结束后保存退出，续跑前移走该文件。强制中断后，应先确认旧训练进程已退出，再处理遗留的 `runs/queue.lock`。

## 评估与结果生成

训练完成后，依次复核最佳检查点、冻结评估协议并执行测试：

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

评估协议固定检查点、测试清单和扰动子集，推理使用 FP32。输出包括逐图预测 CSV、概率 NPZ、指标 JSON、性能测量和分析图，保存在 `results/final/`；导出的推理权重保存到 `weights/leaf_ds_se.pt`。

如果需要为新训练的检查点重新建立评估协议，执行 `python scripts/begin_new_evaluation.py`，程序会先归档已有协议和结果。

## 单图预测

使用训练得到的检查点识别图片：

```powershell
python predict.py --image "D:\photos\leaf.jpg" --checkpoint runs/resnet18_ds_se_s42_224/best.pt
```

完成权重导出后，可使用默认权重及生成的示例图：

```powershell
python predict.py --image assets/examples/example_0.jpg
python predict.py --image "D:\photos\leaf.jpg"
```

终端打印前五个类别及其 softmax 分数，并保存 `results/demo/prediction.json` 和 `results/demo/prediction.png`。可通过 `--device cuda` 使用 GPU，或通过 `--output` 指定输出目录。

## 代码结构

| 文件 | 功能 |
|---|---|
| `src/models.py` | 网络结构、参数量与 MACs 统计 |
| `src/data.py` | 数据读取、增强与归一化 |
| `src/inference.py` | 模型加载、单图推理与 Grad-CAM |
| `train.py` | 训练、验证和断点恢复 |
| `evaluate.py` | 测试集及固定扰动评估 |
| `predict.py` | 命令行单图预测 |
| `configs/formal/` | 六组训练配置 |
| `scripts/` | 数据准备、训练队列、结果复核、计时与绘图脚本 |
