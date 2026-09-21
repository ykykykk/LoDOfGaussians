# Resident v2：增量 SPT、跨视角预取与 CUDA 热点优化

## 启用

`configs/sports_shoes_resident.json` 现在显式使用 `resident_version=2`。原来的 resident 命令更新代码后即可使用 v2；legacy 配置仍不自动切换。也可以在原命令中加入 `--training_backend resident --resident_version 2`。

从 LoDOfGaussians 项目根目录执行，数据必须是照片与内参匹配的去畸变 COLMAP 数据：

```powershell
git switch yk
git pull --ff-only origin yk

$Data = 'D:\Datasets\sports_shoes_colmap_pinhole'
$Output = Join-Path $Data 'output_resident_v2'
& '.\.venv\Scripts\python.exe' train.py `
    --project_dir $Data `
    --config sports_shoes_resident.json `
    --output_dir $Output `
    --resolution 1 `
    --seed 0 `
    --export_ply (Join-Path $Output 'sports_shoes_finest.ply')
```

以上路径是示例，首次验证使用新输出目录。`--resolution 1` 保持原图；速度比较必须保持你原来的分辨率，不能把从 1600 像素宽切换到原图增加的工作量算成后端退化。保留粗训练 6000 次、细训练 20000 次和之前的分裂阈值/间隔，并不预先保证这个迭代数对所有数据已收敛。

## 已接入训练的优化

### 1. 按变化更新 SPT

`IncrementalSPT` 将构建结果按子树缓存。分裂边界先写回 GPU 的最新参数，分块精确比较位置、log-scale 和树连接关系；因此几何更新、跨子树 relocation、父子关系变化都会使相关缓存失效，颜色/透明度更新则不必重算空间范围。

只重算失效子树的距离范围与包围球。子树布局未变且改动较少时直接更新对应 GPU 数组区间；布局改变、改动覆盖多数条目或区间很多时进行一次线性拼装，避免大量小传输。构建过程使用列表收集、一次拼接，避免反复拼接不断增长的数组。上层树范围也会刷新，支持普通叶子、SPT 0 和不同 skybox 数量。

这是**精确增量更新，不是承诺每次只做 O(新增点数) 工作**。仍有 O(全局点数) 的分块变化扫描和上层分区检查；布局改变时仍可能全量线性重排。整个场景都被训练更新时，所有相关子树都必须刷新。CPU 快照约增加每点 44 字节及每点 4 字节的归属表，另有缓存的 SPT 条目；需要相应主内存预算。

### 2. 跨视角流水与有界预取

一个 CPU 后台取图任务提前获取下一视角。图像/相机参数由独立 CUDA copy stream 上传；当前视角继续在训练 stream 渲染与反向。Gaussian 预取使用下一视角的实际切面，只填空槽位或干净的非活跃槽位，保护当前训练点以及下一视角已经命中的点。

固定页中转块由完成事件持有，完成前不会释放或重写。使用预取数据前等待对应事件；保存、分裂、relocation、映射失效前排空传输。跨树修改边界不预取 Gaussian，已准备的相机按新树版本重新选择，避免读取旧点集或旧 Adam 状态。

预取是机会性的：CPU 下一张尚未准备好、容量不足、没有安全槽位或上一次中转仍在用时，正常按需加载，不丢点，也不为了预取强迫脏数据写回。并发能力不等于一定有明显重叠，需在目标 GPU 实测；预取也可能因带宽竞争或低命中率而变慢。

### 3. 索引式 CUDA Adam 和参数 gather

新增小型 PyTorch C++/CUDA 扩展，继续使用原 gsplat 渲染与反向。活跃包只 gather 参数，不再把 Adam m/v 一起复制出来；单个索引 CUDA kernel 直接更新固定槽位里的参数和 m/v。保持 FP32、原有 beta/epsilon/学习率与全局迭代偏差修正，冷点不更新动量。同一点集仍复用活跃包。

扩展支持 Windows MSVC 的 `/O2` 构建参数，CUDA 不启用 fast-math。`native_ops="auto"` 在主训练进程首次使用时构建并缓存；编译失败会明确警告并回退 Torch，同时记录原因。`"cuda"` 为严格模式，失败时报错；`"torch"` 强制参考实现。首次编译时间不计作稳定迭代速度。

### 4. 上层切面 CUDA 化

上层树的可见性/祖先选择条件在 CUDA 中并行求解，再按预计算的参考遍历顺序输出；保留既有 CUDA SPT 切面函数。上层包围范围在树刷新时缓存，避免每个视角重复计算。原 Torch 选择路径仍作为对照。

### 5. 有界图像解码缓存

已解码的 CPU 相机/图像通过 LRU 缓存复用，总配置预算分摊到 DataLoader worker；Windows spawn 不复制已有缓存。相机对象复制后才替换设备字段，缓存中的 CPU 图像不会被改成 GPU 图像。

### 6. 有条件的 CUDA Graph

Torch 后备路径对反复复用的固定活跃包捕获 **Adam 更新段**；动态学习率和偏差修正通过固定地址输入更新。点集改变或树修改时重新捕获/失效。使用索引 CUDA Adam 时，本身已是单 kernel，不重复启用这层图捕获。

这不是整轮动态 gsplat 前向/反向的 CUDA Graph：动态切面及光栅化分配没有强行捕获。没有把参数/Adam 改为半精度，也没有通过额外的微小点/低贡献点裁剪换速度。这些会改变数值或监督，不作为默认的等价提速。

## 配置与内存

| 参数 | 当前默认 | 含义 |
|---|---:|---|
| `incremental_spt` | `true` | 重用未变化的子树 |
| `view_prefetch` | `true` | CPU 下一视角获取与图像/点集机会性预取 |
| `gaussian_prefetch_rows` | `131072` | 预取上限，同时不超过两个 `transfer_rows` 中转块 |
| `image_prefetch_mib` | `512` | 超过预算的图像只在成为当前视角时上传 |
| `image_cache_gib` | `1.0` | 全部 worker 合计的解码缓存预算，0 关闭 |
| `native_ops` | `"auto"` | 自动构建 CUDA 热点扩展，失败有提示地回退 |
| `graph_adam` | `true` | Torch 后备路径的固定包 Adam 图捕获 |
| `graph_min_reuse` | `8` | 连续复用达到此次数才捕获 |
| `pool_gib` / `headroom_gib` | `2.5` / `4.0` | 沿用 v1 的池上限与工作区余量 |

显存预算不等于训练永不 OOM。当前/下一图像、活跃梯度、原 gsplat 的排序/反向工作区另占显存。池放不下当前完整切面时仍走完整点集的按需传输，不降低细节。显存紧张时先降低预取/缓存预算，不需要改变照片分辨率或 PLY 叶节点。

## 结果与 A/B 验证

真实细节阈值分裂、保存前完整写回和最高细节全部叶节点 PLY 导出沿用上一版。没有改变 PLY 的 SH/尺度/透明度编码和 skybox 策略。

独立视角随机数生成器让 v2 的“预取开/关”具有相同的采样次序；原 v1 的随机数消费方式、旧 SPT 相同距离键的非稳定排序与 v2 不保证逐位相同。CUDA 浮点路径也需要容差与最终画质比较。

运行参考配置 `sports_shoes_resident_reference.json` 可在 v2 中关闭增量复用、预取、解码缓存、native 和 graph；它与优化配置有相同的照片/迭代/分裂预算及视角调度。切回整个旧 resident 实现则使用 `--resident_version 1`，其 v2-only 配置会被明确忽略；`--training_backend legacy` 保留最初实现。

```powershell
# 使用项目已有环境；pytest 未安装时先安装 pytest。
& '.\scripts\verify_resident_v2.ps1' -RequireCuda

# 这是 Adam 微基准，不是整轮训练加速倍数。
& '.\.venv\Scripts\python.exe' -m tools.benchmark_resident_v2 --rows 100000

& '.\.venv\Scripts\python.exe' -m tools.summarize_resident_profile `
    (Join-Path $Output 'resident_profile.jsonl') --warmup 100
```

新测试覆盖：脏缓存/预取/淘汰/重载、直接传输、固定槽位、epoch 失效、增量与完整构建、直接执行仓库原有 SPT 构建算法的对照、解码缓存、确定性调度，以及真实 v2 训练控制流的 CPU 替身集成。GPU 测试覆盖 native Adam/预取、上层切面与 Adam Graph；没有 CUDA 时标记跳过，不能算通过。

`resident_run.json` 写入版本、配置、native 是否真正加载与编译失败原因。`resident_profile.jsonl` 延续各阶段计时，并增加预取命中/取消、避免的 m/v gather 字节、SPT 重建/复用/上传、图捕获/重放等计数。CUDA span 含流等待，不能当作纯 kernel 耗时，也不能与 CPU host_ms 简单相加。

以相同数据、分辨率、种子、训练预算比较预热后的整段 wall time、活跃点数、显存峰值和相同相机的渲染质量。实现和 CPU 回归完成不代表 Windows/RTX 3090 已实训验收；本轮不预先承诺倍数。
