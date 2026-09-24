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

先通过梯度阈值筛选叶节点，仅当合格者过多时按评分限制本轮工作量。默认单次分裂父节点不超过当前叶节点的 15%，新增树节点不超过 200000，总节点上限为 6000000。各项均为上限，不是点数目标。classic 不使用 `densify_percent` 控制数量；该参数仅用于 MCMC。classic 子节点沿父高斯最长局部轴对称分开，偏移由原尺度与重定位后尺度的方差差确定，保留中心及该轴二阶矩；LoD 父节点保持原样。这为新增节点提供空间分工，最终画质仍需训练与留出视角评估。

自动步数：细训练为 `clamp(10 × 训练照片数, 20000, 120000)`，粗训练为 `clamp(2 × 训练照片数, 3000, 12000)`。分裂安排约 64 个窗口，最后约 20% 步数优化已有点。显式步数会同步调整学习率和细化节奏；这些是启发式预算，并不是验证误差驱动的自动收敛判断。

## 图像和显存

启动时估计训练图片缓存后的大小。能放入不超过 8 GiB、且不超过当前可用内存 25% 的预算时，使用同进程共享 CPU 图像缓存，通过取图线程预取。这样避免 Windows 多进程反复传递整张图片。同进程缓存将图片锁页一次并在后续视角复用，省去每步重复锁页复制；锁页内存计入既有缓存预算。更大的数据保留分 worker 的有界 LRU 缓存。粗训练也复用解码缓存。

默认 `resident.compact_images=true`：仅当 RGB 或 alpha 张量通过逐值精确往返检查时，缓存并传输 uint8，在 GPU 恢复原来的 FP32 值。没有缩图、重新 JPEG 压缩或降低训练计算精度。带软遮罩的预乘 RGB 等不能精确还原的字段继续使用浮点。恢复时使用设备上的除数，避免 CUDA 标量倒数乘法与 CPU 原目标之间的一位浮点差异。粗训练损失使用相同还原规则。设为 `false` 可对照原浮点图片路径。

预检对无外部遮罩的 RGB 图估计字节缓存，其余格式保守按浮点估计。`decoded_training_bytes` 表示原浮点大小，`cached_training_bytes` 表示计划缓存大小；实际 LRU 始终按真实张量字节计费。运动鞋原图约由 16.18 GB 降至 4.04 GB，因此无需放大默认 8 GiB 上限即可缓存全部训练照片。GPU 仍需要完整 FP32 图片和渲染工作区。

## 更高细节与大图

`scripts/train_general.ps1 -Config sports_shoes_resident.json` 使用更新后的运动鞋预设：归一化梯度、1 像素 LoD、自动图像输入策略、细化诊断及 1200 万总节点上限。默认通用预设仍为 600 万总节点。改变配置后使用新输出目录。

先检查 `densification.jsonl`：有很多超过阈值的候选但分裂受限时，可增加 `densify_max_leaf_fraction` 或 `densify_max_new_nodes`；候选很少时，应检查可见叶节点和评分分布，再调整 NDC 单位的 `densify_grad_threshold`，例如从 0.001 试到 0.0005。仅增加 `cap_max` 对没有合格候选的窗口无效。降低阈值也可能拟合噪声，必须比较留出视角。

`-Resolution 2` 对 6004×4010 图片使用 3002×2005；`-Resolution 1` 保留原图，像素数是前者的四倍，图像存储与逐像素运算增加，但整步耗时不一定恰好四倍。原图中的真实纹理有助于细节，放大模糊图不会恢复信息；对焦、运动模糊、视角覆盖和标定误差仍需从输入照片判断。

优先保留现有 CUDA 运算、图片缓存和预取，在可承受的分辨率上增加真实细化；`-Iterations 30000` 会同步延长学习率和增点日程。若原图无法整体缓存，检查解析出的 `image_io` 和 profile 的 `data_wait`、`render_loss`、`backward`，再决定提高缓存预算还是降低训练尺寸。不要仅凭显存空闲就放大缓存或同时提高所有参数。当前 `-SkipIfExists` 只复用配置匹配的粗模型，不支持从低分辨率细训练结果继续高分辨率训练。

`resident.adaptive_pool=true` 使常驻容量跟随已初始化点数增长。在分裂写回、映射失效的安全边界重新计算容量。`pool_gib` 仍是上限，`headroom_gib` 保留渲染工作区余量；不会因为上限足够大就一开始为全部潜在点数分配显存。

针对本机 RTX 3090，通用和运动鞋预设的常驻池上限调整为 8 GiB，最少保留 6 GiB 工作区余量。分裂边界还会根据已观察到的临时分配峰值增加预留：`max(6 GiB, 1.25 × (历史峰值 allocated - 当前 allocated) + 1 GiB)`。实际容量继续受空闲显存、缓存行数上限和当前点数约束；8 GiB 是高斯池上限，不是总显存预算或预分配目标。其他显卡可调整这两个已有参数。

当前视角超过常驻池容量时，仍保留完整视角点集并回写状态；原生后端可用时，溢出路径也执行融合 CUDA Adam，直接更新 `[参数 | 一阶矩 | 二阶矩]`，减少普通 PyTorch 更新产生的大临时张量。未加载原生后端时保留原有 Torch 路径。

无损图片缓存还会省去完全不透明的 alpha 遮罩。此时目标图不变，损失中的全白遮罩乘法被省略；真正的遮罩和软透明度继续保留。

启用 `compact_images` 时，无外部遮罩的普通 RGB 照片在解码后直接保留字节像素，避免先展开 FP32 再压回字节的 CPU 内存开销。缩放仍采用原来的 PIL LANCZOS；进入损失计算前恢复原有 FP32 像素值。RGBA、外部遮罩、测试视图半幅遮罩继续使用原浮点处理路径。

预算不等于绝不 OOM。活跃图像、梯度、光栅化工作区和其他应用都会占用显存。当前切面超过缓存容量时，保留完整切面按需传输，不删点来凑容量。

Windows 长训练还应检查 `reserved_bytes`：缓存分配可能被 WDDM 放入共享内存，不能把它当成额外物理显存。Resident v2 按物理显存约束常驻池预算，在已同步回写的扩容边界释放旧池缓存；每步开始时，若保留量超过物理显存减 `headroom_gib` 且闲置缓存超过 1 GiB，则回收闲置块。`allocator_trim` 记录回收耗时。此操作不改变分辨率、点数、损失或优化器状态，也不保证单个超大视图不会超过显存预算。

## 诊断与复用

长时间 Windows 训练可用 `scripts/start_background.ps1 -Runner <训练启动脚本.ps1>` 交给当前用户的 Windows 任务计划运行，独立于启动应用的进程生命周期。任务使用普通权限、不保存密码、没有执行时限；需要保持该用户登录。启动脚本应自行重定向日志并记录退出状态。

Resident v2 可设置 `resident.checkpoint_every=1000`，每 1000 步原子更新 `resident_latest.pt`。检查点包含参数、Adam 两阶矩、树结构、增点统计、迭代位置和 Torch 随机状态；保存失败会保留上一份完整文件。恢复使用相同配置、数据、粗模型和随机种子，给 `scripts/train_general.ps1` 传入 `-SkipIfExists -ResumeCheckpoint <resident_latest.pt>`。恢复会跳过已完成的视图采样，不重跑粗训练；目前检查点要求 `vary_distance_multiplier=false`。`.dhier`/PLY 仅为模型导出，不包含完整优化器状态。

继续已完成模型的增点训练时，用新输出目录复用同一粗模型，显式传入 `-AllowGrowthResume`。此选项只允许增加总节点上限、每轮新增节点数、总步数，并调整增点截止步及间隔；数据、粗模型、其余优化参数及 Adam 状态保持一致。新增点窗口从零评分和可见性重新累计。若使用通用策略，设置 `general_policy.preserve_fine_schedule=true`，显式给出延长后的增点截止步和间隔，并保持原 `position_lr_max_steps`，避免恢复时学习率跳升；原始检查点和旧输出继续保留。

`densification.jsonl` 区分实际可见叶节点、正梯度叶节点、超过阈值节点以及实际分裂数量。零梯度不会被误当成不可见；连续三个窗口没有合格节点时警告。日志里的总节点包含 LoD 父级，PLY 只导出最细叶节点，二者应分别检查。

通用模式会保存 `dataset_manifest.json`。`-SkipIfExists` 复用粗训练前检查相机文件、图像/稀疏点元数据、训练分辨率、随机种子和粗训练关键设置。缺失清单或不匹配时使用新输出目录。指纹不是所有图片的全内容哈希，预检也不能证明位姿正确。

## 测试

在已加载 MSVC 的项目终端执行：

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-dev.txt
& '.\.venv\Scripts\python.exe' -m pytest tests/test_general_training.py tests/test_streaming_v2.py tests/test_resident_pool.py tests/test_training_regressions.py -q
```
