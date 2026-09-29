# YK Gaussian — 单卡无 LoD 高斯训练

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

## 来源与许可证

本项目部分代码源自 [A LoD of Gaussians](https://github.com/FelixWindisch/LoDOfGaussians)，并包含其使用的 3D Gaussian Splatting、Hierarchical 3D Gaussians 等组件。原作者署名和适用许可保留，详见 [LICENSE.md](LICENSE.md) 及各组件许可证。项目由 ykykykk 独立维护。
