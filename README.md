# ResNet-18 叶片病害识别

基于 PlantVillage 训练 ResNet-18，并比较深度可分离卷积（DS）、SE 注意力和 DS+SE。四种结构各训练两个种子、每次60轮。推荐的 DS 模型平均测试准确率为99.7105%，参数量较基线减少81.73%。

## 直接预测

使用 Python 3.10，在项目根目录安装依赖。训练环境采用 PyTorch 2.6.0、torchvision 0.21.0 和 CUDA 12.4。

```powershell
git clone https://github.com/ChaiJunyan/plant-leaf-cnn.git
cd plant-leaf-cnn
python -m pip install -r requirements.txt
python predict.py --image assets/examples/example_0.jpg
```

默认使用仓库内的 `weights/leaf_ds.pt`，按 CPU、FP32 推理，输出 top-5，结果保存到 `results/demo/prediction.json` 和 `prediction.png`。可用自己的图片替换 `--image`，用 `--output` 指定保存目录。

比较 DS+SE 时显式指定权重。

```powershell
python predict.py --image assets/examples/example_0.jpg --checkpoint weights/leaf_ds_se.pt
```

## 无需重新训练的结果复核

仓库附有原始预测、概率、训练日志、数据划分清单和冻结协议。仅复核数字时只需安装 NumPy，无需下载数据集或安装 PyTorch。

```powershell
python -m pip install numpy==1.23.5
python scripts/verify_evidence.py
```

脚本校验压缩包内全部文件的 SHA256，从233,728条预测重建92组混淆矩阵和分类指标，并复核双种子统计、训练轮数、12次外部预测和拒识阈值。输出 `status` 为 `passed` 表示通过。[复核材料说明](evidence/README.md)

## 重新训练

完整的数据准备、六组主实验、补充重复和验证对照命令见[复现步骤](docs/reproduce.md)。训练需要支持 BF16 的 CUDA 显卡，检查点按验证集 Macro-F1 选择。

| 结构 | 修改 |
|---|---|
| ResNet-18 | 经典基线，随机初始化 |
| DS | 将 layer3、layer4 的分支3×3卷积替换为 DW+PW |
| SE | 在残差分支末端 BN 后、相加前加入通道注意力 |
| DS+SE | 同时采用上述两项修改 |

代码入口为 `train.py`、`evaluate.py`、`predict.py`，结构实现在 `src/models.py`，数值汇总和复核脚本位于 `scripts/`。

## 数据与实验范围

数据使用 [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage) 固定版本 `9e97599868962bd0079b8db4b7f1efa9185fa1e7`，许可为 [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)。样例图来自验证集，来源记录见 `assets/examples/metadata.json`。推理权重来源及指纹见 `weights/inference_manifest.json`。

文献参照为 Mohanty 等2016年研究摘要报告的99.35%。本实验采用不同划分、224×224输入和随机初始化 ResNet-18，文献采用 GoogLeNet 迁移学习，两者作为任务水平参照。报告保留95%自定工程目标，并以同协议 ResNet-18 作为结构消融基线。[原论文](https://doi.org/10.3389/fpls.2016.01419)

模型适用于训练类别覆盖的清晰单叶图像。外部图片中，两个模型对三张可映射到训练类别的叶片均输出了与来源标注不符的 Top-1。六张探索案例中，DS+SE 全部拒识，DS 接收了三个不符输入内容的输出。这些外部结果不支持 DS 的泛化优势。来源与许可见 `assets/external_review_sources.json`。
