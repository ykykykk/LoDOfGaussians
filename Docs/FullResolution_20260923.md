# 原图无损缓存与实测

原图维持 6004×4010，训练、损失、梯度和模型参数继续使用 FP32。本轮仅优化图像输入，不降低增点阈值、不裁图、不减少视角或点数。

`resident.compact_images=true` 将可精确往返的 FP32 图片字段存为 uint8，上传 GPU 后恢复。每个字段在首次解码后经过精确相等检查；不能无损表示的软遮罩预乘结果等保留浮点。GPU 使用设备张量除法，避免标量倒数乘法的一位浮点误差。粗训练和细训练共享恢复规则。

42 张训练原图的预计缓存从 16.18 GB 降为 4.04 GB，在此次机器可用内存下落入默认 8 GiB 上限，启用同进程缓存与预取。实际预算约 4.25 GB，LRU 按真实张量存储计费。空 masks 目录不再被当作每张图都有遮罩。内存不足或其他图像格式可能仍走 worker LRU。

## 对照结果

Windows、RTX 3090 24 GB、PyTorch 2.14.0+cu126、MSVC 14.44。相同种子 0、相同视角顺序、同一已有的 6000 步粗模型、300 步原图细训练，第 200 步分裂。基线关闭 `compact_images`，使用解析后的 4-worker/2 GiB 浮点缓存；优化组使用字节格式的共享完整缓存。

| 指标 | 浮点输入基线 | 无损字节缓存 |
|---|---:|---:|
| 预热后每步中位耗时 | 364.67 ms | 102.21 ms |
| 整段细训练，含启动/解码/收尾 | 124.14 s | 54.40 s |
| 图片累计上传量 | 115.95 GB | 28.99 GB |
| PyTorch 峰值 allocated | 3878.14 MiB | 3928.55 MiB |
| 最终叶节点 | 67,096 | 67,096 |
| 留出原图 PSNR | 26.12261 dB | 26.12337 dB |
| 留出原图 SSIM | 0.871615 | 0.871653 |

预热段约 3.57 倍、整段约 2.28 倍，上传量减少约 75%。预热指标取 profile 的 100–190、220–290 步之间相邻十步区间的每步耗时中位数，排除第 200 步的增点区间。基线与优化组顺序运行，机器仍运行其他应用；不将短程收益外推为所有数据或完整训练的固定倍数。

输入还原通过全部 256 个字节值的 CPU/GPU 精确相等检查；训练结果不保证逐位一致。单个留出视角 `DSC_2006_0.jpg` 的渲染平均绝对像素差 0.00253、最大差 0.30015，但 PSNR/SSIM 接近。未完成原图的 20000 步收敛质量评估。

124 项回归通过。另从头跑通原图 20 步粗训练 → hierarchy 生成 → 20 步细训练 → 58,677 叶节点 PLY，约 31.96 秒；这是入口检查，不代表画质。

原始结果：`D:\Resources\ProductionResources\Models\Clothes\Sports Shoes\3dgs\alod_artifacts\fullres_audit_20260923`，包含两组模型、`metrics.json`、`performance.json` 及 `cli_smoke`。GPU 仍需容纳完整浮点图像和反向传播工作区，本次峰值显存没有降低。

## 正式原图训练命令

在项目目录的 PowerShell 7 中运行，自动建立新的输出路径：

```powershell
$RunOutput = Join-Path 'D:\Resources\ProductionResources\Models\Clothes\Sports Shoes\3dgs\alod_artifacts' ('fullres_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& '.\scripts\train_general.ps1' `
    -Data 'D:\Resources\ProductionResources\Models\Clothes\Sports Shoes\colmap_pinhole' `
    -Output $RunOutput `
    -Config sports_shoes_resident.json `
    -Resolution 1
```

预设为 6000 步粗训练、20000 步细训练，自动导出 `scene_finest.ply`。需要更长优化时可加 `-Iterations 30000`；这会增加时间，并不保证质量单调改善。细节和增点诊断见 [训练指南](General_Training.md)。
