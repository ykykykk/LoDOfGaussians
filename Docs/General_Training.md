# 通用数据训练

入口为 `configs/general_balanced.json`，面向 Resident v2、classic 和已去畸变的 PINHOLE/SIMPLE_PINHOLE COLMAP 数据。

## 运行

在 PowerShell 7 中，从项目目录运行：

```powershell
& '.\scripts\train_general.ps1' `
    -Data 'D:\Datasets\my_scene_colmap' `
    -Output 'D:\Results\my_scene_01' `
    -Resolution 2 `
    -PlanOnly
```

检查输出目录中的 `dataset_plan.json` 和 `resolved_config.json` 后，去掉 `-PlanOnly` 即开始训练。脚本会加载 MSVC 14.44；可用 `-VcVarsAll` 指定安装位置，用 `-Toolset` 指定已安装工具集。

`-Resolution 2` 表示宽高各减半；`1` 保持原图。通用策略不自动修改这项质量目标。`-Iterations`、`-CoarseIterations` 可显式覆盖步数；0 表示由数据量解析。首次使用新配置请选择新的输出目录。

## 相机、评分和尺度

相机保存训练尺寸下的 `K_train`，渲染使用真实主点。图像缩放后的 fx、fy、cx、cy 与当前图片同步，LoD 也使用当前像素焦距。平方距离判断使用焦距比例的平方。

分裂评分使用 `norm([gx * width/2, gy * height/2])`，继续采用 ALoD 的窗口最大值统计。`0.001` 是归一化最大值评分的初始阈值，不是所有数据的最佳阈值。重采样、纹理、遮罩、曝光仍会影响真实梯度。

SPT 分区阈值由 `SPT_relative_volume * scene_camera_radius^3` 得到。原始坐标、单位、方向以及导出 PLY 坐标保持原样；这不是整套优化器对任意坐标单位的严格数值等价保证。

## 质量预算

先通过梯度阈值筛选叶节点，仅当合格者过多时按评分限制本轮工作量。默认单次分裂父节点不超过当前叶节点的 15%，新增树节点不超过 200000，总节点上限为 6000000。各项均为上限，不是点数目标。子节点生成继续调用 ALoD 原有实现。

自动步数：细训练为 `clamp(10 × 训练照片数, 20000, 120000)`，粗训练为 `clamp(2 × 训练照片数, 3000, 12000)`。分裂安排约 64 个窗口，最后约 20% 步数优化已有点。显式步数会同步调整学习率和细化节奏；这些是启发式预算，并不是验证误差驱动的自动收敛判断。

## 图像和显存

启动时估计训练图片解码后的大小。能放入不超过 8 GiB、且不超过当前可用内存 25% 的预算时，使用同进程共享 CPU 图像缓存，通过取图线程预取。这样避免 Windows 多进程反复传递整张浮点图像。更大的数据保留分 worker 的有界 LRU 缓存。粗训练也复用解码缓存。

`resident.adaptive_pool=true` 使常驻容量跟随已初始化点数增长。在分裂写回、映射失效的安全边界重新计算容量。`pool_gib` 仍是上限，`headroom_gib` 保留渲染工作区余量；不会因为上限足够大就一开始为全部潜在点数分配显存。

预算不等于绝不 OOM。活跃图像、梯度、光栅化工作区和其他应用都会占用显存。当前切面超过缓存容量时，保留完整切面按需传输，不删点来凑容量。

## 诊断与复用

`densification.jsonl` 区分实际可见叶节点、正梯度叶节点、超过阈值节点以及实际分裂数量。零梯度不会被误当成不可见；连续三个窗口没有合格节点时警告。日志里的总节点包含 LoD 父级，PLY 只导出最细叶节点，二者应分别检查。

通用模式会保存 `dataset_manifest.json`。`-SkipIfExists` 复用粗训练前检查相机文件、图像/稀疏点元数据、训练分辨率、随机种子和粗训练关键设置。缺失清单或不匹配时使用新输出目录。指纹不是所有图片的全内容哈希，预检也不能证明位姿正确。

## 测试

在已加载 MSVC 的项目终端执行：

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-dev.txt
& '.\.venv\Scripts\python.exe' -m pytest tests/test_general_training.py tests/test_streaming_v2.py tests/test_resident_pool.py tests/test_training_regressions.py -q
```
