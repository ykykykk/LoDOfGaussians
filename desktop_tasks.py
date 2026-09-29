"""Shared desktop task definitions."""
TASKS = {
    '查看场景': ('view', [('checkpoint', '分页检查点目录', 'dir'), ('port', '预览端口', '8765'), ('pool-gib', '预览缓存 GiB', '4')]),
    '分页训练': ('train-paged', [('checkpoint', '输入分页检查点', 'dir'), ('output-dir', '新输出目录', 'newdir'), ('source-path', '数据目录（可选）', 'dir'), ('config', '配置（可选）', 'json'), ('steps', '本次优化步数', '1000'), ('viewer-port', '预览端口', '8765')]),
    '场景初始训练': ('train', [('project_dir', 'COLMAP 数据目录', 'dir'), ('output_dir', '新输出目录', 'newdir'), ('config', '训练配置', 'json'), ('iterations', '训练步数（可选）', ''), ('coarse_iterations', '粗训练步数（可选）', '')]),
    '转换为分页检查点': ('convert-blocks', [('input', 'Flat 检查点 .pt', 'pt'), ('output', '新输出目录', 'newdir')]),
    '迁移为 Flat': ('flat', [('input', 'Resident 检查点 .pt', 'pt'), ('output', '新检查点 .pt', 'savept'), ('config', 'Flat 配置', 'json')]),
    '导出分页 PLY': ('export-ply', [('checkpoint', '分页检查点目录', 'dir'), ('output', '输出 PLY（可选）', 'saveply')]),
    '导出 Resident PLY': ('flat', [('input', 'Resident 检查点 .pt', 'pt'), ('output', '输出 PLY', 'saveply')]),
    '质量评估': ('evaluate', [('checkpoint', '分页检查点目录', 'dir'), ('output-json', '评估结果 JSON', 'savejson'), ('camera-limit', '评估相机数量', '5')]),
    '环境诊断': ('doctor', []),
}
