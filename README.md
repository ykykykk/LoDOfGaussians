# A LoD of Gaussians — Windows 单卡增强分支（yk）

这是基于 [FelixWindisch/LoDOfGaussians](https://github.com/FelixWindisch/LoDOfGaussians) 上游历史维护的个人派生版本，面向 **Windows、单张 NVIDIA GPU、大场景、原尺寸照片和高点数训练**。原论文、算法与官方实现的作者归属属于上游作者；本仓库不是官方发布。

- 个人仓库 / `origin`：[ykykykk/LoDOfGaussians](https://github.com/ykykykk/LoDOfGaussians)，修改发布在 `yk` 分支。
- 上游仓库 / `upstream`：[FelixWindisch/LoDOfGaussians](https://github.com/FelixWindisch/LoDOfGaussians)，本地 `main` 跟踪 `upstream/main`。
- 正式称呼是 **派生仓库 + upstream remote 工作流**。本仓库目前是独立 GitHub 仓库，不是 GitHub Fork 网络内标记的 fork，但保留上游提交历史，仍可通过 Git 获取、比较及合并上游更新。
- [上游原始 README](#上游原始-readme) 保留在本文后半部分；许可证见 [LICENSE.md](LICENSE.md)，本分支未更换上游许可证。

## 修改与功能

| 模块 | 本分支的修改 | 使用意义 |
| --- | --- | --- |
| Windows 环境 | PowerShell 7 启动器、MSVC/CUDA 构建适配、UTF-8 编译输出 | 在 Windows 运行原生 CUDA 路径 |
| Resident v2 | CPU 保存全局模型与 Adam 状态，GPU 固定槽位常驻池、活跃参数复用、异步预取 | 支持模型大于常驻池的单卡训练 |
| CUDA 热点 | 索引参数 gather、融合 Adam、上层切面选择、溢出路径融合更新 | 减少临时张量及传输，保留 FP32 |
| 空间层级 | 按变化刷新 SPT，布局变化时批量重建 | 复用未变化的子树；仍可能全局扫描或重排 |
| 图片输入 | 有界 LRU、线程取图、锁页复用、RGB 直接字节解码、可精确还原的 uint8 缓存 | 保留原图目标值，减少缓存与上传开销 |
| 相机与细节 | 缩放后真实内参/主点、像素一致 LoD、归一化梯度评分、classic 对称分裂 | 修正尺度与细节分裂，并限制单轮增点量 |
| 显存管理 | 自适应常驻池、物理显存预算上限、扩容及压力下回收闲置缓存 | 抑制长训练中的缓存累积和共享内存压力 |
| 恢复训练 | 原子保存完整检查点，恢复参数、Adam、树、细化统计与采样进度 | 从最近保存步继续 |
| 后台运行 | Windows 任务计划启动器 | 独立于启动终端/应用的进程生命周期 |
| 导出与诊断 | 最细叶节点 PLY、解析配置、数据清单、增点与性能 JSONL | 避免叠加导出 LoD 父节点，记录真实增长和瓶颈 |

上游 `legacy` 后端仍保留，新通用入口使用 Resident v2。Resident v2 要求 CPU backing、classic 细化，训练损失路径为 RGB；深度监督等不支持项应使用兼容的 legacy 配置。`native_ops=auto` 在扩展不可用时可回退 Torch，`cuda` 要求原生后端成功。CUDA Graph 仅覆盖部分固定包 Adam 路径，不是整轮动态渲染捕获。

详解：[通用训练](Docs/General_Training.md) · [Resident v2](Docs/Resident_Training_v2.md) · [单卡大点数](Docs/SingleCard_20260923.md) · [原图缓存](Docs/FullResolution_20260923.md) · [细节修复](Docs/Detail_Optimization_20260922.md)

## 获取代码与 Windows 安装

需要 Git、PowerShell 7、`uv`、NVIDIA 驱动、CUDA Toolkit 12.6（`nvcc` 可用）、Visual Studio C++ 构建工具，以及 CMake/Ninja。项目使用 Python 3.10，本机训练使用 MSVC 14.44。安装脚本会安装依赖并编译扩展；依赖没有全部锁定，首次编译可能耗时较长。

```powershell
git clone --recursive --branch yk https://github.com/ykykykk/LoDOfGaussians.git
cd LoDOfGaussians
git remote add upstream https://github.com/FelixWindisch/LoDOfGaussians.git
git fetch upstream
git config remote.pushDefault origin
git config branch.yk.pushRemote origin
& .\scripts\setup_windows.ps1
```

已有 checkout 可运行 `git submodule update --init --recursive`。已有 `upstream` 时先用 `git remote -v` 查看，不要重复添加。remote 属于本地配置，普通 clone 不会继承，因此新机器需执行添加步骤。

训练脚本默认使用 `.venv\Scripts\python.exe`，自动定位 Visual Studio 并加载 MSVC 14.44；其他位置/工具集可通过 `-VcVarsAll`、`-Toolset` 指定。单独调用 Python 导出、评估或测试时，使用同一已配置的 Visual Studio x64 开发终端和项目 Python。

## 数据格式

使用已去畸变、照片与标定一致的 COLMAP 数据（PINHOLE / SIMPLE_PINHOLE）：

```text
my_scene/
├── images/
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

相机/图像记录也支持对应 `.txt`；稀疏点支持 `.bin`、`.txt` 或已有 `points3D.ply`。可选输入及上游流程见后半部分原始说明；Resident v2 不支持深度监督。输出建议放在数据旁的独立目录。

## 预检与完整训练

以下路径均为示例，替换成自己的数据和输出位置。在仓库根目录执行，新配置第一次训练使用新的输出目录。

基于通用预设创建配置，显式启用每千步检查点（通用预设未默认开启检查点）：

```powershell
New-Item -ItemType Directory -Force 'D:\Results' | Out-Null
$Config = 'D:\Results\my_scene_config.json'
$cfg = Get-Content .\configs\general_balanced.json -Raw -Encoding utf8 | ConvertFrom-Json -AsHashtable
$cfg.resident.checkpoint_every = 1000
$cfg | ConvertTo-Json -Depth 20 | Set-Content $Config -Encoding utf8

$Run = @{
    Data = 'D:\Datasets\my_scene'
    Output = 'D:\Results\my_scene_01'
    Config = $Config
    Resolution = 1
    Iterations = 30000
    CoarseIterations = 6000
    Seed = 0
}
& .\scripts\train_general.ps1 @Run -PlanOnly
# 查看输出目录的 dataset_plan.json 与 resolved_config.json 后正式运行：
& .\scripts\train_general.ps1 @Run
```

`Resolution=1` 使用原图，`2` 为宽高各减半。不指定显式步数时，通用策略按训练视图数估算：粗训练 3000–12000 步，精训练 20000–120000 步。这是启发式预算，不是收敛保证。

训练完成自动导出 `scene_finest.ply`。层级文件名由 `output_file_name` 决定，通用配置为 `scene.dhier`。

### 大场景、更多点和显存参数

在首次运行前修改上述配置；从完整检查点恢复要求训练配置保持一致。

| 参数 | 通用预设 | 含义 |
| --- | --- | --- |
| `cap_max` | 6000000 | 全部树节点上限，包含父级，不等于最终 PLY 点数 |
| `cache_size` | 8000000 | 缓存行数上限，还受点数和显存预算约束 |
| `densify_grad_threshold` | 0.001 | NDC 评分阈值；降低可增加合格候选，也可能增加噪声 |
| `densify_max_new_nodes` | 200000 | 单窗口新增树节点上限 |
| `densify_max_leaf_fraction` | 0.15 | 单窗口允许分裂的叶节点比例上限 |
| `resident.pool_gib` | 8 | 高斯常驻池预算，不是总显存目标 |
| `resident.headroom_gib` | 6 | 为图像、渲染和反传预留的最少空间 |
| `resident.adaptive_pool` | true | 按已初始化点数调整容量 |
| `resident.compact_images` | true | 可精确还原的图片使用字节缓存和传输 |
| `resident.checkpoint_every` | 未配置时为 0 | 0 关闭；1000 为每千步保存 |

大场景可显式设 `cap_max=20000000`、`cache_size=20000000`，但这是容量上限，不保证长到该数量。先看 `densification.jsonl` 中的合格候选、实际增长和上限，再调整阈值/预算。清晰度、视角覆盖、位姿误差仍会影响细节；不要以占满共享显存为目标。

## 中断后的续训

沿用上述 `$Run`，或重新填写相同数据、配置、输出、分辨率、步数和种子：

```powershell
& .\scripts\train_general.ps1 @Run -SkipIfExists `
    -ResumeCheckpoint (Join-Path $Run.Output 'resident_latest.pt')
```

`-SkipIfExists` 单独使用仅复用配置匹配的粗模型；精训练恢复还需 `-ResumeCheckpoint`。目前检查点要求 `vary_distance_multiplier=false`。PLY / `.dhier` 不含完整 Adam 状态，不能替代检查点。

完整检查点在最终迭代之前按间隔保存，所以完成 30000 步时最新 `.pt` 可能仍为 29000 步；最终 `.dhier` 和 PLY 包含完成时模型。强制停止会失去上次检查点之后尚未保存的迭代。

## 独立后台运行、查看和停止

将下列内容保存为 `D:\Results\run_my_scene.ps1`，替换仓库、数据和配置路径。配置应已开启检查点。

```powershell
#requires -Version 7.0
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$Run = @{
    Data = 'D:\Datasets\my_scene'
    Output = 'D:\Results\my_scene_01'
    Config = 'D:\Results\my_scene_config.json'
    Resolution = 1
    Iterations = 30000
    CoarseIterations = 6000
    Seed = 0
    SkipIfExists = $true
}
$checkpoint = Join-Path $Run.Output 'resident_latest.pt'
if (Test-Path $checkpoint) { $Run.ResumeCheckpoint = $checkpoint }
$log = Join-Path $PSScriptRoot ('training_' + (Get-Date -Format yyyyMMdd_HHmmss) + '.log')
try {
    & 'D:\Code\LoDOfGaussians\scripts\train_general.ps1' @Run *> $log
} catch {
    $_ | Out-String | Add-Content $log -Encoding utf8
    exit 1
}
```

从仓库根目录启动并保存任务名：

```powershell
$task = & .\scripts\start_background.ps1 -Runner 'D:\Results\run_my_scene.ps1'
$task | ConvertTo-Json | Set-Content 'D:\Results\my_scene_task.json' -Encoding utf8
Get-ScheduledTask -TaskName $task.TaskName
Get-Content 'D:\Results\training_实际时间.log' -Tail 20 -Wait
```

后台任务需要当前 Windows 用户保持登录。后续管理同一个任务，避免重复调用启动脚本创建并发训练：

```powershell
$task = Get-Content 'D:\Results\my_scene_task.json' -Raw -Encoding utf8 | ConvertFrom-Json
Get-ScheduledTask -TaskName $task.TaskName
# 按需强制停止，未保存的迭代会丢失：
Stop-ScheduledTask -TaskName $task.TaskName
# 确认任务及训练子进程退出后，按需恢复同一个任务：
# Start-ScheduledTask -TaskName $task.TaskName
```

`Ready` 也可能表示失败或被停止，应结合实际进程、日志完成/报错信息和输出文件判断；仅凭 `Running` 也不能确认迭代仍在推进。

## PLY、结果与诊断

完整训练自动生成 `scene_finest.ply`；从已有层级重新导出：

```powershell
& .\.venv\Scripts\python.exe -m tools.export_ply `
    --input 'D:\Results\my_scene_01\scene.dhier' `
    --output 'D:\Results\my_scene_01\scene_finest_reexport.ply'
```

默认仅导出最细叶节点，排除父级和 skybox；`--include-skybox` 可保留背景。强制停止后可从完整检查点导出：将以下代码保存为仓库根目录下的 `export_saved_checkpoint.py`，替换检查点路径。

```python
from pathlib import Path
import torch
from tools.export_ply import export_gaussian_tensors_ply, finest_leaf_indices
source = Path(r'D:\Results\my_scene_01\resident_latest.pt')
s = torch.load(source, weights_only=True, map_location='cpu', mmap=True)
p = s['properties']
width = p.shape[1] // 3
output = source.parent / f"scene_finest_step{s['iteration']}.ply"
if output.exists():
    raise FileExistsError(output)
export_gaussian_tensors_ply(p[:, :3], p[:, 10:13], p[:, 14:width],
    p[:, 13:14], p[:, 3:6], p[:, 6:10], output,
    indices=finest_leaf_indices(s['nodes']))
print(output)
```

```powershell
& .\.venv\Scripts\python.exe .\export_saved_checkpoint.py
```

评估或交互查看应使用与训练一致的配置和分辨率。评估工具从 `configs/` 读取配置，因此先为本次结果放入一份具名配置（示例采用新的 `my_scene_eval.json`）：

```powershell
Copy-Item 'D:\Results\my_scene_01\resolved_config.json' '.\configs\my_scene_eval.json'
& .\.venv\Scripts\python.exe eval_hierarchy.py `
    --hierarchy 'D:\Results\my_scene_01\scene.dhier' `
    -s 'D:\Datasets\my_scene' --resolution 1 --config my_scene_eval.json
& .\.venv\Scripts\python.exe hierarchy_viewer.py `
    --hierarchy 'D:\Results\my_scene_01\scene.dhier' `
    -s 'D:\Datasets\my_scene' --resolution 1 --config my_scene_eval.json
```

| 输出 | 用途 |
| --- | --- |
| `scene_finest.ply` | 完整训练结束后的最细叶高斯模型 |
| `*.dhier` | LoD 层级模型，用于项目查看器/评估 |
| `resident_latest.pt` | 最近原子保存的完整精训练检查点 |
| `scaffold/point_cloud/iteration_*/` | 粗模型及层级 |
| `resolved_config.json` / `dataset_plan.json` | 实际配置及输入预算 |
| `resident_run.json` | 后端、设备及实际运行选项 |
| `resident_profile.jsonl` | 数据等待、切面准备、渲染、反传、Adam、重建和显存采样 |
| `densification.jsonl` | 可见叶节点、评分候选及真实增点 |

续训 JSONL 按运行追加，回退到检查点后可能有重复迭代记录；统计增长和速度需按运行段或有效恢复轨迹处理。画质仍需看渲染和留出视角指标，不能仅凭 loss、点数或 PLY 完整性判断。评估/查看参见 [Viewing](Docs/Viewing.md) 和下方上游命令。

## 已验证记录与边界

2026-09 单张 RTX 3090 DJI 场景记录：原图 8236×5474，407 个注册视图，其中 402 个训练视图；复用 6000 步粗模型，完成 30000 步精训练。该场景设为 2000 万总节点上限，最终 19,999,999 个层级节点，PLY 9,950,000 个叶节点（1,034,800,633 字节）。原始数据、模型和训练日志保存在本地，不作为代码仓库内容发布。

缓存修复后的两段完整续训中，采样 `reserved` 最高约 18.8 GiB；最后 15000 步约 3 小时 45 分钟，平均约 0.90 秒/步。这是特定硬件、数据和配置的记录，不是跨硬件性能保证，也不是严格的新旧代码 A/B。最终文件已核对结构与点数，尚未作视觉质量验收。

代码 `0983642` 对应回归记录为 108 项通过。需要自行验证时，在已加载 MSVC 的项目开发终端执行：

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
& .\.venv\Scripts\python.exe -m pytest tests/test_resident_pool.py tests/test_general_training.py tests/test_streaming_v2.py tests/test_resident_checkpoint.py -q
```

## 上游同步与个人发布

| 名称 | 目标 | 用法 |
| --- | --- | --- |
| `origin` | 个人 GitHub 仓库 | 发布 `yk` 分支 |
| `upstream` | 原作者仓库 | 获取、比较和合并更新；保留真实 fetch/push URL |
| 本地 `main` | 跟踪 `upstream/main` | 保持上游基线 |
| 本地 `yk` | 跟踪 `origin/yk` | 开发、训练与个人修改 |

新 clone 如需建立上游基线且还没有本地 `main`：

```powershell
git branch --track main upstream/main
```

同步上游前保存自己的工作并确保工作区干净：

```powershell
git fetch upstream
git switch main
git merge --ff-only upstream/main
git switch yk
git merge main
git submodule update --init --recursive
```

如有冲突，解决、提交并验证后再发布；取消正在进行的合并可使用 `git merge --abort`。这不是强制重置，不会要求丢弃个人修改。

发布个人修改：

```powershell
git switch yk
git status --short
git add README.md  # 换成实际需要提交的文件，避免加入数据和模型
git commit -m "Describe the change"
git push origin yk
```

推荐 `git config remote.pushDefault origin` 和 `git config branch.yk.pushRemote origin`，使默认推送去个人仓库；显式推送仍写清 `origin yk`。这些设置不移除 `upstream` 推送 URL，也不授予原库写权限。本分支维护流程不执行上游推送。未来只有获得原库写权限并明确决定贡献时才考虑向其推送；是否使用 GitHub Fork/PR 取决于上游贡献流程。

以下保留本地上游基线 `73f547341547ae021ff036ad74fdf73a1240b450` 的原始 README，其中“官方实现”描述上游作者版本。使用本增强分支优先参考上面的 Windows/Resident v2 指令。

---

## 上游原始 README

<h1 align="center">A LoD of Gaussians: Out-of-Core Training and Rendering for Seamless
Ultra-Large Scene Reconstruction</h1>

<p align="center">
  <a href="https://felixwindisch.github.io/ALoDOfGaussians/">
    <img src="https://img.shields.io/badge/Project-Page-darkblue" alt="Project Page">
  </a>
  <a href="https://arxiv.org/abs/2507.01110">
    <img src="https://img.shields.io/badge/arXiv-2603.24725-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://cloud.tugraz.at/index.php/s/tRz85cJsRQGJX4q">
    <img src="https://img.shields.io/badge/Data-Uni10k-darkorange" alt="Point Clouds">
  </a>
  <a href="https://youtu.be/5mRpZGSqoyg">
    <img src="https://img.shields.io/badge/Video-YouTube-red" alt="Video">
  </a>
</p>

<h3 align="center">SIGGRAPH 2026</h3>

<h4 align="center">
    <a href="https://felixwindisch.github.io/">Felix Windisch</a><sup>1</sup> ·
    <a href="https://derthomy.github.io/">Thomas Köhler</a><sup>1</sup> ·
    <a href="https://r4dl.github.io/">Lukas Radl</a><sup>1</sup> ·
    <a href="https://mattiadurso.github.io/">Mattia D'Urso</a><sup>1</sup> ·
    <a href="https://steimich96.github.io/">Michael Steiner</a><sup>1</sup> ·
    <a href="https://schmalstieg.github.io/">Dieter Schmalstieg</a><sup>1</sup> ·
    <a href="https://www.markussteinberger.net/">Markus Steinberger</a><sup>1,2</sup>
</h4>

  <div align="center">
    <p>
      <sup>1</sup> Graz University of Technology 🇦🇹<br>
      <sup>2</sup> Huawei Technologies 🇦🇹
    </p>
  </div>

## Overview
**A LoD of Gaussians** enables seamless ultra-large 3DGS training and rendering on consumer GPUs through a combination of out-of-core streaming and level of detail.
This repository contains the official authors' implementation associated with the paper "A LoD of Gaussians: Unified Training and Rendering for Ultra-Large-Scale Reconstruction with External Memory". 
## Setup

Make sure to clone the repository using `--recursive`:
```
git clone git@github.com:FelixWindisch/LoDOfGaussians.git --recursive
cd LoDOfGaussians
```

Setting up the conda environment:
```
conda create -n LoDOfGaussians
conda activate LoDOfGaussians
conda install python=3.10
conda install -c nvidia cuda-toolkit=12.6
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
pip install submodules/simple-knn --no-build-isolation
pip install submodules/gaussianhierarchy --no-build-isolation
pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation

```


### Compiling hierarchy generator and merger
These files were adapted from Hierarchical 3DGS and can be built as follows:
```
cd submodules/gaussianhierarchy
cmake . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --config Release
cd ../..
```
## Running the method

#### Dataset 
Prepare your dataset in the standard 3DGS format:
```
root/
├─ sparse/
│  ├─ 0/
│  │  ├─ cameras.bin
│  │  ├─ images.txt
│  │  ├─ points3D.txt
├─ images/
├─ masks/
├─ depths/
```
If depth images or masks are used, place them in root/depths and root/masks respectively.
To start training, execute:
```
python train.py --project_dir root --config default.json --skip_if_exists
```
When training for the first time, gsplat takes a few minutes to initialize. If this initialization fails, try manually setting the CUDA paths:
```
export CUDA_HOME=$CONDA_PREFIX

export CPATH=$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

We provide basic hyperparameters in configs for basic large-scale scenes (default.json), smaller test scenes (small.json) and the MatrixCity dataset.
Check out the [Hyperparameter Guide](Docs/Hyperparameters.md) to create your own configuration file tailored for your scene.

Training runs in 2 steps: Coarse Optimization (Standard 3DGS, sparse point cloud) and Fine Optimization (Out of Core and LoD, with densification). 

After finishing both steps, a _out.dhier file will be written to ```root/outputs```, which can be rendered and evaluated:
```
python eval_hierarchy.py --hierarchy /path/to/result.dhier -s root/  --config default.json
python hierarchy_viewer.py --hierarchy /path/to/result.dhier -s root/  --config default.json
```
```eval_hierarchy``` will render all images in the test set (use the llffhold in your config parameter to designate every nth image for testing) and output quality metrics.
```hierarchy_viewer``` allows interactive viewing of the results. This can be done using the networked inria viewer, but we strongly recommend installing SplatViz (https://github.com/Florian-Barthel/splatviz) and running it with ```python run_main.py --mode=attach``` while ```hierarchy_viewer``` is running. Check out [Docs/Viewing.md](Docs/Viewing.md) for additional viewer features.

### Disclaimer
Note that this code release version relies on the gsplat rasterizer and will thus be more memory-efficient than reported in the paper.

