# codex_augmentation

一个基于 Tkinter 的 PCB 缺陷合成工具（MVP）：

- 源图上点选 SAM 提示点（可选）或使用简化自动分割
- 自动轮廓精修（去小噪声、闭运算、平滑）
- 贴到目标图后支持 `alpha` / `poisson-normal` / `poisson-mixed` 融合
- 贴片几何增强：缩放、旋转、翻转（X/Y）
- 贴片颜色匹配：按目标局部统计自动匹配亮度/对比度
- 可选后处理和谐化：`Harmonize=pctnet`（当前为边界抑制适配层，可替换为官方 PCT-Net 推理）
- 批量合成：从目标图目录随机采样并自动导出
- 导出合成图 + COCO 标注（bbox + segmentation）

## 快速开始

### Windows

先安装 Anaconda 或 Miniconda，然后运行：

```bat
git clone https://github.com/Dave7922/codex_augmentation.git
cd codex_augmentation
run_windows.bat
```

`run_windows.bat` 会使用 Conda 环境 `pcb311`。如果本机还没有该环境，脚本会自动创建：

```bat
conda create -y -n pcb311 python=3.11
```

后续更新：

```bat
cd codex_augmentation
git pull
run_windows.bat
```

### macOS / Linux

```bash
cd codex_augmentation
conda create -y -n pcb311 python=3.11
conda activate pcb311
pip install -r requirements.txt
python app.py
```

## SAM 使用（可选）

默认可在无 SAM 环境下运行（使用 OpenCV 的快速分割回退）。

若要启用 SAM：

1. Windows 用户运行 `run_windows.bat`，脚本会在 `pcb311` 中安装 CPU 版 PyTorch 和 `segment-anything`
2. 脚本会自动下载 SAM checkpoint 到 `checkpoints/sam_vit_b_01ec64.pth`
3. 脚本会自动下载 PCT-Net 权重到 `third_party/PCT-Net-Image-Harmonization-main/pretrained_models/PCTNet_ViT.pth`
4. 启动后会自动加载默认 SAM checkpoint；也可以点击 `Load SAM Checkpoint` 手动选择

如果手动安装，请确保在 `pcb311` 环境内执行：

```bash
conda activate pcb311
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install git+https://github.com/facebookresearch/segment-anything.git
mkdir checkpoints
mkdir -p third_party/PCT-Net-Image-Harmonization-main/pretrained_models
curl -L -o checkpoints/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
curl -L -o third_party/PCT-Net-Image-Harmonization-main/pretrained_models/PCTNet_ViT.pth https://github.com/rakutentech/PCT-Net-Image-Harmonization/raw/main/pretrained_models/PCTNet_ViT.pth
```

## 交互说明

- `Load Source`: 加载带缺陷的源图
- 在左图点击：添加正样本点（前景）
- `Shift + 左键`：添加负样本点（背景）
- `Segment`: 生成缺陷 mask
- `Refine Mask`: 自动轮廓精修
- `Extract Patch`: 按当前 mask 提取缺陷贴片
- `Load Target`: 加载目标 PCB 图
- 在右图点击：设置粘贴中心
- `Synthesize`: 执行融合合成
- `Batch Synthesize`: 批量模式，选择目标图目录与输出目录后自动生成
- `Export COCO`: 导出 `output/` 下图像与 JSON

### 参数建议

- `Scale`: 建议 `0.7~1.4` 区间优先，避免偏离真实尺寸分布
- `Rotation`: 可先在 `-30~30` 内使用，再逐步扩大
- `Color Match`: 推荐保持开启，提高贴片与局部背景一致性
- `Harmonize`: `off` 或 `pctnet`（建议先配合 `poisson-mixed` 使用）
- `Batch N`: 批量样本数量

## 注意

- Poisson 融合要求 mask 质量较好，建议先 `Refine Mask`
- MVP 版本默认单实例导出；批量生成可在此基础上扩展
