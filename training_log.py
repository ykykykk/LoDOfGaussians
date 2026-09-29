"""Human-readable summaries for structured training records."""
def summary(record, english=False):
    if not all(key in record for key in ('iteration', 'loss', 'tile_steps_per_second')):
        return None
    step = int(record['iteration'])
    coverage = float(record.get('image_equivalent_progress', step))
    loss = float(record['loss'])
    speed = float(record['tile_steps_per_second'])
    points = int(record.get('total_points', 0))
    memory = float(record.get('peak_allocated_bytes', 0)) / 2**30
    if english:
        line = f'Update {step:,}  |  Image-equivalent {coverage:,.1f}  |  Loss {loss:.5f}  |  {speed:.1f} updates/s  |  {points:,} points  |  Peak GPU {memory:.2f} GiB'
    else:
        line = f'更新 {step:,} 步  |  全图等效 {coverage:,.1f} 步  |  损失 {loss:.5f}  |  {speed:.1f} 步/秒  |  {points:,} 点  |  峰值显存 {memory:.2f} GiB'
    if record.get('checkpoint'):
        line += '  |  Checkpoint saved' if english else '  |  检查点已保存'
    if record.get('growth'):
        line += '  |  Densification' if english else '  |  已执行增点'
    return line
