# Windows：按细节分裂、训练提速与最高细节 PLY

## 本次行为

`classic` 分裂恢复上游阈值选择：仅分裂 `_densification_criterium > densify_grad_threshold` 的叶节点，不再用低分节点补足固定增长配额。保留此前 Windows 兼容与单节点距离返回值修复。

细训练的驻留数据分成当前活跃 Gaussian 和未参与当前渲染的 GPU 缓存 Gaussian。两部分的属性、Adam 一阶/二阶状态及细节梯度现在会在分裂前完整写回 CPU；保存中间或最终 `.dhier` 前也会写回。分裂评分不再依赖节点是否刚好被缓存淘汰，保存不再遗漏驻留参数的最新更新。

PLY 默认只导出 `child_count == 0` 的全部叶节点。这是自适应树的最细完整切面，包含不同树深度的叶节点，不是只取全树最大 depth。内部父级 LoD 不导出；独立 skybox 默认排除，可单独导出时加 `--include-skybox`。没有按大小、透明度或相机视锥额外删减叶节点。

导出保留 `.dhier` 的 log-scale 与 wxyz 旋转，将已激活 opacity 转回 logit，并把 SH 转为标准 PLY 的通道优先顺序。按块选取叶节点写出，避免先复制整份最高细节模型；写出成功后再替换目标文件，失败不会留下覆盖了旧结果的半成品。

## 速度修改

- 删除细训练每一步两次无条件 `torch.cuda.empty_cache()` 和一次无条件 `torch.cuda.synchronize()`；及时释放重排临时张量，交给 PyTorch 缓存分配器复用。分裂阶段仍保留必要的阶段性回收。
- 删除细训练中随后被覆盖的额外 SSIM 计算，保留上游实际生效的损失公式；合并重复 loss 标量读取，删除只有 `pass` 的全数组 NaN 扫描。发现非有限 loss 时明确报错，不继续污染训练。
- 开销较大的缓存调试检查由 `pipe.debug` 控制。正常训练不再每步执行这些调试归约/去重。
- 粗训练复用项目已有的 fused-SSIM，提供 JSON 参数 `coarse_fused_ssim=false` 回退参考实现。数学目标相同，但浮点数不保证逐位一致；回归测试中提供 CUDA 下的值与梯度比较。
- 相机数据通过 DataLoader 的自定义 `pin_memory()` 支持固定页内存，图像与相机张量使用非阻塞上传。`data_workers`、`data_prefetch_factor`、`pin_memory` 可配置；`data_workers=0` 也有效。关闭 DataLoader 时不再为关闭操作额外启动一组 worker。

没有更换 Gaussian 渲染器或 Adam，没有降低 SH 阶数或自动降低训练照片分辨率。没有新增强制点数增长。

## 鞋子质量配置

使用 `configs/sports_shoes_quality.json`，命令中 `--config` 只填写文件名。

保留粗训练 6000 次、细训练 20000 次。分裂间隔改为 500，阈值为 `5e-6`，在第 500 次之后开始、16000 次之前结束，给新增节点留下后续优化时间。这是减少过密全树重建的起点，不保证该迭代数对所有数据都已收敛。

`vary_distance_multiplier=false`：不再随机把相机当成更远距离训练，优先近距离细节。

`clear_cache_interval=0`：关闭固定间隔清空 GPU 数据缓存；达到 `cache_size` 后的容量淘汰仍保留。`cache_size=8000000`、`cache_size_after_reduction=6000000` 没有放大。缓存分配器的 reserved 显存可能提高，这不等于新增活跃 Gaussian，也不构成永不 OOM 的保证。

首次验证建议使用新输出目录。示例中的数据路径需要替换为实际的、已经去畸变且相机参数匹配的 COLMAP 数据目录。

```powershell
$Data = 'D:\Datasets\sports_shoes'
$Output = Join-Path $Data 'output_detail'
$Python = '.\.venv\Scripts\python.exe'

& $Python train.py `
    --project_dir $Data `
    --config sports_shoes_quality.json `
    --output_dir $Output `
    --resolution 1 `
    --export_ply (Join-Path $Output 'sports_shoes_finest.ply')
```

`--resolution 1` 保持原图尺寸。上游默认 `-1` 会把较宽照片缩到约 1600 像素宽；速度 A/B 对比必须使用相同分辨率，不能把切换到原图增加的计算量算成代码变慢。

## 直接重导已有结果

已有 `.dhier` 无需重训即可去除父级重叠：

```powershell
& '.\.venv\Scripts\python.exe' -m tools.export_ply `
    --input 'D:\Results\sports_shoes_quality.dhier' `
    --output 'D:\Results\sports_shoes_finest.ply'
```

重导只能使用文件里已经保存的数据；旧版本曾留在 GPU、没有写入 `.dhier` 的更新无法通过重导恢复。要验证驻留写回和细节分裂修复，需要重新训练。

## 验证与性能边界

```powershell
& '.\.venv\Scripts\python.exe' -m pip install pytest
& '.\.venv\Scripts\python.exe' -m pytest tests/test_training_regressions.py -q
```

测试覆盖叶节点选择、SH 0–3 阶字段、透明度/尺度/旋转、分块与原子写出、活跃/缓存属性及 Adam/梯度写回、DataLoader 关闭行为，以及训练调用位置。CUDA fused-SSIM 一致性测试在没有 CUDA 时跳过。

CPU 回归和语法检查不能代替 Windows/CUDA 实训。需用同一数据、相同分辨率与相近活跃点数对比预热后的迭代耗时；另外记录分裂/重建耗时、总训练时间、显存峰值及同相机渲染质量。修复评分后可能生成更多真正需要的点，后期单步耗时也可能随有效点数增加。

本次不预先承诺加速倍数；总体收益取决于瓶颈是图像加载、缓存重排、SSIM、全树重建还是 Gaussian 光栅化。
