# YK Gaussian Studio

面向 Windows 的 Gaussian Splatting 精细模型训练工具。使用同一个 `YK-Gaussian.exe` 打开桌面界面或执行 CLI；人工操作和 AI 本地控制共用项目状态。

## 快速开始

从 [Releases](https://github.com/ykykykk/LoDOfGaussians/releases) 下载对应版本的**全部分卷**、`Extract-Portable.cmd` 和 `SHA256SUMS.txt`，放在同一目录，双击解压脚本。解压后运行 `YK-Gaussian.exe`，移动时保留整个目录结构。

- Windows x64，兼容的 NVIDIA 显卡和驱动。
- 便携版包含 Python、PyTorch、CUDA 运行库和 Qt，无需另装 Python 或 CUDA Toolkit。
- CUDA 扩展在 RTX 3090（计算能力 8.6）上验证；其他 GPU 可先运行 `YK-Gaussian.exe doctor` 检查。
- 界面根据系统语言选择中文或英文。
- 软件输入是已经配准、去畸变的 COLMAP 数据，暂不执行照片配准。

## 数据布局

```text
dataset/
├─ images/
│  ├─ photo_001.jpg
│  └─ photo_002.jpg
├─ masks/                 # 可选，与照片同名、同尺寸
│  ├─ photo_001.png
│  └─ photo_002.png
└─ sparse/0/
   ├─ cameras.bin
   ├─ images.bin
   └─ points3D.bin
```

支持对应的 COLMAP 文本格式。独立 mask 的白色代表主体、黑色代表背景，灰色保留软边缘。未提供独立 masks 时，可使用照片自身的 Alpha 通道；显式 masks 优先。项目文件 `.ykproject.json` 引用数据与结果路径，不内嵌大型训练文件。

## 项目流程

| 步骤 | 用途 |
|---|---|
| 01 导入数据 | 选择数据、输出目录、Mask 模式，可导入已有检查点 |
| 02 检查数据 | 检查相机、照片、尺寸和遮罩 |
| 03 构建初始模型 | 粗训练（scaffold）及初始 Resident 训练 |
| 04 准备精细模型 | 转换为可用于精细训练的分页模型 |
| 05 训练与细化 | 调整学习率、增点日程、点数上限及显存配置 |
| 06 检查与评估 | 留出视角的图像质量评估 |
| 07 导出成果 | 导出 Gaussian PLY；裁剪模式同时保存独立裁剪结果 |

可以提前选中后续步骤调整参数；依赖尚未完成时，运行按钮不可用。支持单步运行和“一键到底”，遇到错误会停止。改变影响模型的前置参数会清空相关步骤状态，但不删除已有磁盘成果；仅影响显示或保存频率的设置不会使训练结果失效。

所有下拉框禁用滚轮切换，避免滚动参数面板时误操作；仍支持点击和键盘选择。

## Mask 三种模式

在 **01 导入数据 → Mask 处理** 中选择：

| 模式 | 训练行为 |
|---|---|
| 不用 Mask (`none`) | 使用完整 RGB 图像，忽略独立遮罩及照片透明度 |
| 忽略背景 (`ignore`，默认) | 遮罩外不提供直接图像损失梯度，不因 mask 删除三维点 |
| 裁剪 (`crop`) | 监督渲染透明度匹配 mask，并定期剔除多视角确认的背景点 |

裁剪模式的透明度监督用于改善轮廓：主体内部要求覆盖、外部要求透明，灰色边缘保留连续过渡。RGB 目标与渲染使用相同背景合成，让越过轮廓的高斯收到梯度并调整位置、尺度和不透明度。这个目标与 [Brush 的透明图像模式](https://github.com/ArthurBrussee/brush#training) 相同，但实现与优化器不同，不保证相同画质。

几何删除并非“把球透明度设成 0 再删除”。软件按多视角投票，并考虑高斯支撑范围和边界余量，保留证据不足或靠近轮廓的点：

- 粗训练开始前裁剪，之后默认每 250 次更新及结束时检查。
- Resident 层级阶段继承已裁剪的粗模型，不在父子树中直接删行；准备精细模型时再次裁剪。
- 分页精细训练开始时、每 250 次更新及结束时清理，并同步参数、Adam 状态、增点分数与缓存。
- 导出时复核，生成独立裁剪检查点、点数报告和 PLY。

切换三种模式会使数据检查及初始训练之后的步骤需要重新执行，旧成果文件保留。Mask 质量和多视角一致性决定可达到的边缘效果。裁剪有额外开销，删除背景后的速度收益取决于场景。

## 保存和继续训练

工具栏“保存”、Ctrl+S 或 `control save`：空闲时保存项目；训练时请求当前更新结束后写入检查点。**看到“训练检查点已保存”和步数后，才表示保存成功。**

重新打开项目，在参数和依赖兼容时继续当前步骤。项目保存会检测磁盘版本，防止旧窗口状态覆盖较新的进度。训练模式区分“继续最新结果”“从准备好的模型重新训练”和“复用导入模型并跳过训练”。

| 文件 | 用途 |
|---|---|
| `scaffold/scaffold_latest.pt` | 粗训练续训检查点 |
| `resident_latest.pt` | 初始 Resident 续训检查点 |
| `blocks/manifest.json` | 精细分页检查点入口，须保留其所在完整目录 |
| `.ply` | 模型交换和查看，不包含完整优化器状态 |

强制停止会丢失最近检查点之后的更新。粗训练恢复会重新建立照片加载顺序，不保证与未中断运行逐位一致。

## 预览

内嵌交互视口支持旋转、平移、缩放和复位，支持 Y/Z 向上，默认 Z。关闭“启用预览”可停止新渲染及快照更新。

默认预览使用完整点数。训练快照最多每秒更新一次，拖动时使用较低预览分辨率、目标 30 FPS，静止时 2 FPS；这些设置不改变训练或导出精度。实际帧率取决于 GPU 负载，不保证开启或关闭预览都会产生明显速度差异。

## GUI、CLI 与 AI 控制

双击 EXE 打开 GUI。GUI 底部命令栏和外部 `control` 命令操作同一个窗口，返回状态遵守同一套参数锁与步骤依赖。

```powershell
.\YK-Gaussian.exe control status
.\YK-Gaussian.exe control open --project "D:\Projects\scene.ykproject.json"
.\YK-Gaussian.exe control set --step initial --key config.coarse_iterations --value 1500
.\YK-Gaussian.exe control preview off
.\YK-Gaussian.exe control run --step initial
.\YK-Gaussian.exe control run-all
.\YK-Gaussian.exe control save
.\YK-Gaussian.exe control stop
```

GUI 命令栏直接输入 `status`、`save`、`run-all` 等，不需要 EXE 和 `control` 前缀。独立批处理入口如 `train-paged`、`view`、`export-ply`、`evaluate` 仍可使用；如需与 GUI 同步，请使用 `control`。

## 配置和步数

训练默认值与 UI 分离，便携版配置位于 `app/configs/`：

| 文件 | 内容 |
|---|---|
| `workflow_defaults.json` | 各步骤默认训练参数；已有项目保留各自配置 |
| `preview.json` | 预览帧率、分辨率和快照设置 |
| `mask_crop.json` | 裁剪频率、投票门槛和边界余量 |
| `mask_loss.json` | 透明度监督权重 |

默认使用原图、粗训练 1500 步、初始训练 1500 步，精细日程累计 8000 全图等效步，6000 后停止增点。它们是通用起点，不是每个数据集的最优参数。

分页训练区分裁块更新数和全图等效覆盖量。日志 `iteration`、CLI `--steps` 和检查点间隔按裁块更新计数；学习率和增点日程按全图等效覆盖量推进。项目 `train.steps=0` 表示运行至完整日程终点。GPU 缓存预算与渲染显存需求不同；SSD 分页不能消除单次渲染显存限制。

## 从源码运行

源码构建需要 Python 3.10、NVIDIA 驱动、CUDA Toolkit 12.6、Visual Studio C++ 构建工具、CMake/Ninja、Git 和 uv。

```powershell
git clone --branch main https://github.com/ykykykk/LoDOfGaussians.git
cd LoDOfGaussians
.\scripts\setup_windows.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
.\.venv\Scripts\python.exe yk_gaussian.py ui
```

原生 CUDA 组件构建需使用正确配置的 MSVC/CUDA 环境。独立后端、迁移及技术记录见 [Resident v2](Docs/Resident_Training_v2.md)、[空间块训练](Docs/NoLoD_Spatial_Plan.md)、[质量与吞吐对照](Docs/Paged_Quality_and_Throughput.md) 和 [通用训练](Docs/General_Training.md)；这些文档可能包含历史实验配置。

## 验证范围

已完成 Mask 透明度梯度检查、粗训练与分页训练短跑、裁剪后保存续训、项目状态保护和下拉框滚轮行为检查。未完成本版本的长时间全流程质量评测，也未验证全部 NVIDIA GPU 架构。

## 来源与许可证

项目由 ykykykk 独立维护，部分代码源自 [A LoD of Gaussians](https://github.com/FelixWindisch/LoDOfGaussians)，并包含 3D Gaussian Splatting、Hierarchical 3D Gaussians 等组件。原作者署名与适用许可保留，详见 [LICENSE.md](LICENSE.md) 及各组件许可证。
