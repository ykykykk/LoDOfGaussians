# YK Gaussian — 单卡精细模型训练

本项目使用 `configs/dji_flat_90m.json` 的 `resident.representation="flat"`：只保留实际高斯点和独立背景点，分裂时两个子点替换原点，不保留父高斯及其 Adam 状态。GPU 位置/半径镜像在 Adam 更新后同步，只用于视锥裁剪，不作 LoD 替换或数量截断。

此配置同时启用 `resident.flat_direct=true` 和 `resident.flat_native=true`：模型在显存预算内时，全部参数及 Adam 常驻 GPU，点编号直接作为槽位，省去逐步 CPU 索引规划；超过预算则回退原流式缓存。融合 CUDA 内核执行视锥裁剪及边界更新，保留 FP32、原图和原损失。关闭这两个选项可使用原路径做对比；它们不改变检查点格式。模型整体驻留不代表单视角渲染一定不会超显存，仍预留渲染空间。

从已有 SH1 Resident v2 检查点迁移，原文件保持不变：

```powershell
.venv/Scripts/python.exe tools/flat_checkpoint.py migrate --input OLD/resident_latest.pt --output NEW/flat_initial.pt --config configs/dji_flat_90m.json
```

使用相同数据、分辨率及源断点对应的 scaffold/数据清单，按原入口传入 flat 配置和 `--skip_if_exists --resume_checkpoint NEW/flat_initial.pt`。迁移保留叶子参数、Adam、步数和 RNG，重新开始增点评分窗口；两种表示的断点禁止隐式混用。

配置中 `cap_max=90000000` 是实际存储点数上限（含背景），`densify_max_new_nodes=600000` 是每窗口**净增**点数，对应旧模式新增 120 万节点后净增 60 万叶子的强度。上限不保证达到，梯度与增点截止步决定最终数量。

最终保存 `resident_latest.pt`；`--export_ply` 导出无父层 PLY，不生成 `.dhier`。中途停止后可用：

```powershell
.venv/Scripts/python.exe tools/flat_checkpoint.py export --input NEW/resident_latest.pt --output NEW/scene.ply
```

当前支持已有 SH1 检查点迁移；新数据仍需原流程产生初始模型后迁移，没有新增无 scaffold 的从零入口。空间裁剪采用分块逐点扫描，单视角可见点过多仍可能超过显存；SSD/虚拟内存不能消除渲染显存限制。这里没有用降分辨率或丢弃可见点规避这一限制。

## SSD 空间块训练（可选大模型后端）

`train_paged.py` 将全局参数和 Adam 放在独立磁盘块中，GPU 只保留固定预算页缓存和块元数据。转换时按高斯支撑半径分组、组内空间排序，避免少量大高斯扩大所有块的边界。训练按原像素裁块，保留 SSIM halo，使用融合 CUDA 页裁剪、索引 Adam 和页包围盒归约；不会生成 LoD 父点。

```powershell
.venv/Scripts/python.exe tools/convert_block_checkpoint.py --input OLD/resident_latest.pt --output NEW/initial
.venv/Scripts/python.exe train_paged.py --checkpoint NEW/initial --output-dir NEW/run01 --config configs/dji_flat_blocks.json --steps 100
.venv/Scripts/python.exe train_paged.py --checkpoint NEW/run01 --output-dir NEW/run02 --steps 100
.venv/Scripts/python.exe tools/export_block_ply.py --checkpoint NEW/run02
```

输入必须为已迁移的 SH1 flat 断点。每次训练派生独立输出目录，保留源断点；新格式使用 `manifest.json`，不能传给旧 `--resume_checkpoint`。同盘文件以硬链接复用，不可变块写新版本；原子提交保留当前与上一代，导出 PLY 默认排除背景。

配置继承断点的训练选项，包括增点截止步；仅提高上限不会重新开启已结束的分裂。`pool_gib` 默认 8 GiB，另留 6 GiB 渲染空间；显式 `capacity_rows` 可覆盖。超出单裁块缓存容量会报告所需容量，不能靠删除可见点继续。断点恢复会校验裁块采样参数，保持每个相机的覆盖进度。

日志的 `iteration` 和 CLI `--steps` 是**裁块优化步**，2048 上限对 DJI 原图形成 15 个面积近似相等的裁块。`image_equivalent_progress` 则累积每块核心面积/原图面积：学习率日程、增点间隔/截止和配置 `iterations` 使用此覆盖量；Adam 偏差修正及 `checkpoint_every` 仍使用优化步数。恢复旧裁块断点时，按已保存相机访问序列重建覆盖量，保留参数及 Adam。新的覆盖量日程不会倒退或重做已完成的参数更新，但不能撤销旧日程造成的历史影响。

两线程解码各自独占有界缓存，保持原相机顺序；增点默认保留 GPU 页，读取最新状态并在检查点统一写盘。`decode_workers=1`、`growth_backend="flush"` 可用于原执行方式对照。纯精修关闭增点时，缓存不再为未来点数预留无用页面，给整图渲染留出显存。

每步处理像素比整图少，不能直接比较两者步/秒；相同像素覆盖量也不保证 Adam 更新或收敛结果相同。当前约 2000 万点仍优先使用上面的全显存快路径；分页后端用于突破总模型驻留限制，尚未替代默认入口。测试、实施计划及容量边界见 [空间块实施记录](Docs/NoLoD_Spatial_Plan.md) 和 [质量与吞吐对照](Docs/Paged_Quality_and_Throughput.md)。

固定留出视角原生像素评估（只读断点，可输出中心 1024 像素细节图）：

```powershell
.venv/Scripts/python.exe tools/evaluate_block_quality.py --checkpoint NEW/run02 --output-json NEW/quality.json --camera-limit 5 --preview-dir NEW/previews
```

## 实时可视化（项目内）

独立浏览现有分页检查点，在已配置 MSVC/CUDA 的项目环境运行：

```powershell
.venv/Scripts/python.exe realtime_viewer.py --checkpoint NEW/run02 --pool-gib 4
```

打开终端打印的 `http://127.0.0.1:8765`。左键旋转，右键或 Shift 拖动平移，滚轮缩放；支持复位、预览分辨率选择及持续更新。当前入口接受分页目录或 `manifest.json`，不直接加载 PLY 或 Resident `.pt`。独立查看请使用已经完成的检查点目录，避免训练清理旧块时发生读取冲突。

训练时增加 `--viewer`，可用 `--viewer-port` 修改端口：

```powershell
.venv/Scripts/python.exe train_paged.py --checkpoint NEW/initial --output-dir NEW/run03 --viewer
```

训练预览读取当前 GPU 页及磁盘块，在优化步之间响应请求，不等待检查点；无浏览器请求时不渲染。预览会占用渲染时间、显存，并可能引发页换入换出；界面的分辨率仅影响预览，不改变训练。超出缓存容量时界面显示错误，不丢弃可见点。训练退出后预览服务关闭，可用独立入口继续浏览。

`utils/realtime_viewer.py` 的 `PagedViewRenderer.render()` 与 HTTP/JPEG 界面分离；后续应用可直接调用渲染器，或嵌入本地网页。当前未修改应用构建流程。

## 安装

需要 PowerShell 7、Git、uv、NVIDIA 驱动、CUDA Toolkit 12.6 和 Visual Studio C++ 构建工具，以及 CMake/Ninja；Python 使用 3.10。

```powershell
git clone --branch main https://github.com/ykykykk/LoDOfGaussians.git
cd LoDOfGaussians
& .\scripts\setup_windows.ps1
```

安装脚本创建 `.venv` 并安装、编译依赖。训练入口使用项目虚拟环境；单独执行 Python 工具时应使用配置好 MSVC/CUDA 的 x64 开发终端。

## 数据与初始模型

输入为已去畸变且照片与标定一致的 COLMAP 数据，包含 `images/` 和 `sparse/0/` 下的相机、图像及稀疏点记录。建议将训练输出保存在独立目录。

新数据的初始模型准备见 [通用训练](Docs/General_Training.md)。该流程包含历史层级训练路径；无 LoD 训练使用本文前述 flat 迁移和对应配置，不应将通用层级配置直接当作 flat 配置。

## 检查点与文档

- `resident_latest.pt`：Resident 完整训练检查点，续训需使用与其匹配的数据、配置和表示类型。
- `manifest.json`：SSD 分页训练检查点入口，使用 `train_paged.py --checkpoint` 继续训练。
- PLY 为导出结果，不包含完整优化器状态，不能替代训练检查点。
- [Resident v2](Docs/Resident_Training_v2.md) · [原图缓存](Docs/FullResolution_20260923.md) · [空间块实现](Docs/NoLoD_Spatial_Plan.md) · [质量与吞吐对照](Docs/Paged_Quality_and_Throughput.md)

## Windows 便携应用

### 按项目分步处理

界面按系统语言自动使用中文或英文（非中文系统使用英文），“精细模型”是界面中的统一称呼。

新建或打开 `.ykproject.json` 项目，按左侧步骤推进：**导入数据 → 检查数据 → 构建初始模型 → 准备精细模型 → 训练与细化 → 检查与评估 → 导出成果**。每一步都有基础与高级参数、完成状态和结果路径；保存后可以重新打开继续。修改前置参数会将后续结果标记为需要更新，原有结果文件保留。

可以调节初始训练步数与分辨率、学习率、增点阈值与日程、点数上限、GPU 缓存、渲染预留、图像裁块、保存间隔和评估参数。参数由实际后端执行，不只是界面显示。导入已有检查点时继承其训练约束；已有分页模型保留磁盘块布局，续训采样配置仍须与检查点一致。

初始模型步骤执行 scaffold 与初始 Resident 训练；如果已导入检查点则直接复用。训练每次派生新目录，支持从该项目最新完成结果续训；增加本次优化步数不会自动延长总训练日程，达到终点时需调整训练日程。评估可按需执行，训练完成后也可直接导出。

项目输入仍需已有的已去畸变 COLMAP 数据，本应用暂不执行照片配准。项目文件引用数据和结果路径，不会把大型数据嵌入项目文件。

便携版统一入口为 `YK-Gaussian.exe`：双击打开 GUI，带参数执行 CLI。GUI 底部可输入控制命令；外部终端与 AI 通过 `control` 操作同一个已打开的 GUI。

Qt 深色工作区包含顶部工具栏、左侧场景与工作流、中央原生交互视口、右侧任务参数，以及底部控制台。面板支持停靠与浮动。场景渲染直接显示在应用内，无需打开浏览器；左键旋转、中键或 Shift 拖动平移、滚轮缩放、F 复位。支持分页训练、初始训练、检查点转换、PLY 导出、质量评估和环境诊断。预览端口默认避开占用；停止或退出运行中的任务会确认，尚未保存的训练进度可能丢失。

```powershell
.\YK-Gaussian.exe view --checkpoint 'D:\Results\paged_checkpoint' --pool-gib 4
.\YK-Gaussian.exe ui
```

移动应用时复制整个便携目录。内置 Python 和运行依赖；仍需要兼容的 NVIDIA 驱动。当前 CUDA 扩展在 RTX 3090 上验证，其他 GPU 需运行 `doctor` 检查。

## 来源与许可证

本项目部分代码源自 [A LoD of Gaussians](https://github.com/FelixWindisch/LoDOfGaussians)，并包含其使用的 3D Gaussian Splatting、Hierarchical 3D Gaussians 等组件。原作者署名和适用许可保留，详见 [LICENSE.md](LICENSE.md) 及各组件许可证。项目由 ykykykk 独立维护。

### 工作流默认配置

所有步骤的默认参数集中在 `configs/workflow_defaults.json`；便携版位置为 `app/configs/workflow_defaults.json`。修改后创建的新项目会加载新值，已有 `.ykproject.json` 保存各自参数，不会被覆盖。UI 不定义训练默认值。

默认质量/短日程：原图分辨率，粗训练 1500 步、初始训练 1500 步；精细阶段累计 8000 全图等效步，6000 后停止增点，位置学习率同步衰减至 8000。`train.steps=0` 表示执行完整日程，正数表示本次裁块更新上限。实际裁块更新次数可能大于全图等效步数。这是通用起点，尚未按每个数据集验证最佳质量。

### 统一 GUI / CLI 控制

双击 `YK-Gaussian.exe`，底部命令栏支持 `status`、`preview off`、`preview on`、`run --step initial`、`stop`、`help`。外部终端或 AI 使用相同入口：

```powershell
.\YK-Gaussian.exe control status
.\YK-Gaussian.exe control open --project "F:\Projects\scene.ykproject.json"
.\YK-Gaussian.exe control preview off
.\YK-Gaussian.exe control set --step initial --key config.coarse_iterations --value 1500
.\YK-Gaussian.exe control run --step initial
```

控制命令返回 JSON，并遵守 GUI 的步骤依赖和运行中参数锁。`stop` 会停止当前任务，检查点之后未保存的进度会丢失。原有独立批处理命令仍可使用；如需与 GUI 同步，使用 `control`。

### 保存与恢复训练

工具栏“保存”和 Ctrl+S：空闲时保存项目；训练时请求当前优化步结束后写入训练检查点。界面显示“训练检查点已保存”和步数后才表示已落盘。CLI 可用 `YK-Gaussian.exe control save` 发起同一请求。粗训练默认每 500 步保存 `scaffold/scaffold_latest.pt`，初始 Resident 和精细分页训练按各自检查点间隔保存。中断后重新打开项目，参数和依赖一致时点击“继续当前步骤”即可恢复。恢复粗训练会重建照片随机加载顺序，不保证与未中断运行逐位一致。旧版本没有写入检查点的进度无法恢复。

### 交互预览性能

训练预览使用独立的有限点数模型快照，视角交互不再等待训练步完成；快照更新与画面更新频率分开。默认使用完整点数（`snapshot_points: 0` 表示不限点数），每秒至多更新一次训练快照，拖动时目标 30 FPS / 640 像素宽，静止时 2 FPS。停止拖动后恢复所选预览分辨率。预览不抽样减少模型点数，不改变训练、检查点或导出精度；独立打开检查点仍使用完整渲染。实际帧率取决于 GPU 负载。配置在 `configs/preview.json`，便携版对应 `app/configs/preview.json`。关闭“启用预览”会停止新渲染和快照更新。

### Mask 输入

支持数据集内的 `images/name.jpg` 与 `masks/name.png` 同名配对。Mask 白色保留、黑色忽略，尺寸必须与原图一致；存在 masks 目录时，所有已注册相机必须有对应遮罩。初始训练、精细训练和评估均应用遮罩。没有 masks 目录时保持普通 RGB 训练。
