# Windows Resident Pool v1：减少整池重排，保留真实细节

## 本轮范围

本轮优先实现可独立测试的固定槽位缓存与活跃包复用，仍使用项目现有 gsplat 渲染/反向传播、ALoD hierarchy/SPT 与 classic 分裂。没有引入新的 C++/CUDA 编译依赖。已有配置默认仍使用 legacy；新配置 `sports_shoes_resident.json` 显式启用 resident，也可用 `--training_backend resident` 覆盖旧配置。

数据流现在是：CPU 完整属性/Adam 数据库 → 缺页换入固定 GPU slot → 当前视角紧凑包 → 原有渲染与反向 → 向量化 Adam 更新。冷缓存不参与每轮整池拼接；父级与子级仍通过当前相机的层级切面选择。

## 已实现的优化

1. **固定槽位**：参数、Adam 一阶/二阶状态、分裂评分都有对应的驻留位置。换视角时只加载缺失点；槽位不足时淘汰不在当前请求中的冷点。采用分段环形近似 LRU，避免为少量缺页每次排序整池。
2. **活跃包复用**：gsplat 仍需要紧凑输入，因此并非完全零 gather。点集变化时仅整理当前活跃点；点集与顺序不变时直接复用活跃包与 Adam 状态。在 CUDA 侧先比较 ID，此时也不回读整份 ID 列表。
3. **合并参数组更新**：将同一批 Gaussian 的参数/m/v 整理成统一布局，使用与上游 OurAdam 相同的 beta、epsilon、学习率及全局 iteration 偏差修正。它是向量化 PyTorch 更新，不是新写的 fused CUDA kernel；不为未参与该视角的冷点更新或衰减动量。
4. **有界传输**：最多固定页锁定一个 `transfer_rows` 大小的 CPU 中转块，不锁定整个大场景。覆盖中转块前等待对应 CUDA event；CPU 消费 GPU 写回结果前完成同步。本版没有跨视角异步预取，不宣称 H2D 与训练已经完全重叠。
5. **Windows 导入开销**：`utils/reloc_utils.py` 改为第一次实际 relocation 时才创建 CUDA 组合数表。数值与原版相同；用一次整块上传替代模块导入时 1326 次 GPU 标量写入，避免每个新建的 Windows worker 都在这里初始化 CUDA 工作。

另外，新路径区分 `first_child == 0` 的合法 SPT 0 与普通叶节点。未进入 SPT 的终止粗节点仍参与渲染；视角没有 SPT 时也不会制造一个假的 SPT 或无限跳过训练。SPT 使用当前距离直接求切面，不沿用 legacy 的近似距离容差缓存切面，因此两条完整训练轨迹不保证逐位一致。

## 正确性与显存约束

- classic 分裂继续使用上游梯度阈值，不补足固定点数配额。
- 保存或修改树之前，先完整写回活跃包和脏缓存的参数、Adam 状态与评分，再使映射失效；新子节点不会误用旧槽位。
- PLY 沿用已修正的最高细节全部叶节点导出，包含不同树深度的叶节点，内部父级 LoD 不叠加。独立 skybox 的导出策略与之前相同。
- `pool_gib` 是**常驻池上限**，不是总训练显存。还需要活跃包、梯度、图像和光栅化临时内存；分配容量同时受 `cache_size` 与启动时可用显存估计（驱动空闲量加 PyTorch 可复用保留块）减 `headroom_gib` 限制。
- 当前视角的点集大于池容量时，释放池并使用完整活跃点集直接传输，不降低 LoD 或删除点来凑容量。这种情况下收益可能变小；真实光栅化工作集仍可能 OOM，预算不是永不 OOM 的保证。
- 使用 FP32；不降低 SH 阶数、输入照片分辨率或额外启用小半径/低贡献点裁剪。
- v1 面向当前的 classic + CPU backing + RGB 配置。实验性 MCMC、曝光、深度监督、scale damping、prune-unused 等模式继续使用 legacy。

本轮保留分裂时的全量 SPT 重建：即使没有新增节点，位置/尺度改变也可能需要刷新范围和包围信息。增量 SPT、跨视角预取、自定义索引 CUDA Adam、半精度和 CUDA Graph 必须分别验证，不同时混入这一版本。

## 运行

在项目根目录更新 `yk`，使用已经去畸变且照片/相机匹配的 COLMAP 数据。下面的路径需替换为实际目录：

```powershell
$Data = 'D:\Datasets\sports_shoes_colmap_pinhole'
$Output = Join-Path $Data 'output_resident'
& '.\.venv\Scripts\python.exe' train.py `
    --project_dir $Data `
    --config sports_shoes_resident.json `
    --output_dir $Output `
    --resolution 1 `
    --seed 0 `
    --export_ply (Join-Path $Output 'sports_shoes_finest.ply')
```

6000 次粗训练、20000 次细训练、分裂阈值/间隔与上一版质量配置相同。`--resolution 1` 保留原图；原本用 1600 像素宽或 `--resolution 4` 的训练，速度对比时必须保持原有分辨率，不能把新增像素工作量算成后端变慢。

只切回前一版细训练实现：加 `--training_backend legacy`。它是 A/B 对照与兼容路径，不覆盖新模型或原有 Git 历史。首次训练使用新输出目录；已有 `.dhier` 仍可通过 `python -m tools.export_ply` 单独重导叶节点 PLY。

## 验证与计时

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_resident_pool.py tests/test_training_regressions.py -q
& '.\.venv\Scripts\python.exe' -m tools.summarize_resident_profile `
    (Join-Path $Output 'resident_profile.jsonl') --warmup 100
```

新增测试包括：固定槽位/活跃包复用、冷点动量不变、脏数据淘汰/重载、容量不足直接传输、保存/分裂前写回、同一全局迭代的 Adam 参考一致性、SPT 0/粗节点分类、lazy 组合数表，以及真实训练控制流的 CPU 替身集成测试。CUDA 可用时另测固定页中转复用和真实 gsplat 前向/梯度一致性。CPU 替身测试不等于实际 CUDA 训练。

`resident_run.json` 记录运行配置/设备；`resident_profile.jsonl` 默认每 100 次采样 data_wait、camera_upload、hierarchy_cut、cache_prepare、render_loss、backward、adam、densify_rebuild、save，以及命中、搬运、重排、溢出次数和显存。`host_ms` 是 CPU 侧耗时；`cuda_span_ms` 是 CUDA event 跨度，会包含流等待/CPU 供给间隙，不能把两者直接相加或当成纯 kernel 占比。退出时才等待尚未完成的采样，不每阶段强制同步。

预热后的整段 wall time/iteration 用于比较，分裂/保存耗时单独看。先统一数据、分辨率、种子和训练预算，再报告活跃点数与最终画质；总场景点数多并不意味着某帧的渲染开销相同。当前没有 Windows/RTX 3090 实训加速倍数，不能预先承诺性能提升。

### 下一步由计时决定

`cache_prepare` 占比高：评估索引式 CUDA gather/Adam，进一步去掉活跃包 m/v 搬运。`data_wait` 高：增加受内存预算约束的图像解码缓存/预取。`densify_rebuild` 高：单独实现带回归验证的增量 SPT。`render_loss/backward` 高：当前已主要受光栅化和像素工作量限制，继续改 CPU 缓存未必有大收益。

参考实现接口：gsplat `rasterization`；PyTorch 官方 pin_memory/non_blocking 指南与 CUDA event 语义。固定页中转不能在异步拷贝完成前被 CPU 重写。
