"""Dataset-aware budgets and detail diagnostics; thresholds remain quality gates."""
import copy
import math
import warnings
import torch


def resolve_plan(config, dataset, available_ram):
    out = copy.deepcopy(config)
    policy = out.get('general_policy', {})
    if not policy.get('enabled', False):
        return out
    n = int(dataset['training_views'])
    if n <= 0:
        raise ValueError('No training views after the evaluation split')
    def budget(key, multiple, lo, hi):
        value = int(out.get(key, 0))
        if value < 0:
            raise ValueError(key + ' cannot be negative')
        return value or min(hi, max(lo, multiple * n))
    out['iterations'] = budget('iterations', 10, 20000, 120000)
    out['coarse_iterations'] = budget('coarse_iterations', 2, 3000, 12000)
    steps = out['iterations']
    if steps < 10 or out['coarse_iterations'] < 2:
        raise ValueError('Fine budget must be >=10 and coarse budget >=2')
    if not policy.get('preserve_fine_schedule', False):
        out['position_lr_max_steps'] = steps
        out['densify_from_iter'] = min(500, max(1, steps // 20))
        out['densify_until_iter'] = max(out['densify_from_iter'] + 1, int(steps * .8))
        span = out['densify_until_iter'] - out['densify_from_iter']
        out['densification_interval'] = max(1, math.ceil(span / 64))
    requested = int(out.get('data_workers', 4))
    runtime = out.setdefault('resident', {})
    # Full CPU caching on small datasets avoids Windows IPC of whole float images.
    limit = min(int(policy.get('image_cache_max_gib', 8) * 2**30), int(available_ram * .25))
    decoded = int(dataset['decoded_training_bytes'])
    compact_images = runtime.get('compact_images', True)
    cached = int(dataset.get('compact_training_bytes', decoded)) if compact_images else decoded
    out['coarse_compact_images'] = compact_images
    full_cache = 0 < cached <= limit and policy.get('auto_image_io', True)
    if full_cache:
        out['data_workers'] = 0
        out['pin_memory'] = False  # DMA staging is owned by CameraTransfer.
        runtime['image_cache_gib'] = min(limit, cached * 1.05 + 2**20) / 2**30
    else:
        out['data_workers'] = max(0, requested)
        runtime['image_cache_gib'] = min(limit, 2 * 2**30) / 2**30
    out['resolved_policy'] = dict(training_views=n, estimated_epochs=steps/n,
        decoded_training_bytes=decoded, image_io='shared_cpu_cache' if full_cache else 'worker_lru',
        cached_training_bytes=cached, compact_images=compact_images,
        cache_budget_bytes=int(runtime['image_cache_gib'] * 2**30),
        note='Budgets are bounded heuristics, not a convergence or quality guarantee')
    return out


def relative_spt_volume(opt, radius):
    ratio = float(getattr(opt, 'SPT_relative_volume', 0.0))
    if ratio < 0 or not math.isfinite(ratio):
        raise ValueError('SPT_relative_volume must be finite and nonnegative')
    if ratio:
        if radius <= 0 or not math.isfinite(float(radius)):
            raise ValueError('Degenerate scene camera extent')
        return ratio * float(radius) ** 3
    return opt.SPT_root_volume


def eligible_parents(scores, leaves, threshold, capacity, max_fraction=0., max_new_nodes=0):
    if not math.isfinite(threshold) or threshold <= 0 or not 0 <= max_fraction <= 1:
        raise ValueError('Invalid detail threshold or per-window fraction')
    if max_new_nodes < 0 or capacity < 0:
        raise ValueError('Invalid new-node budget')
    values = scores[leaves]
    if not torch.isfinite(values).all():
        raise FloatingPointError('Non-finite detail scores; inspect training gradients')
    candidates = leaves[values > threshold]
    k = min(len(candidates), capacity // 2)
    if max_fraction:
        k = min(k, math.ceil(len(leaves) * max_fraction))
    if max_new_nodes:
        k = min(k, max_new_nodes // 2)
    if k < len(candidates):
        candidates = candidates[torch.topk(scores[candidates], k, sorted=False).indices] if k else candidates[:0]
    return candidates


class DetailWindow:
    """One visibility bit per global row, including genuinely zero gradients."""
    def __init__(self, capacity, device):
        self.seen = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.views = 0

    @torch.no_grad()
    def observe(self, packet, packed):
        ids = getattr(packet, '_detail_global_ids', None)
        if ids is None:
            ids = torch.as_tensor(packet.ids, device=self.seen.device, dtype=torch.long)
            packet._detail_global_ids = ids
        self.seen.index_fill_(0, ids[packed.long()], True)
        self.views += 1

    @torch.no_grad()
    def report(self, g, threshold, iteration):
        leaf = g.nodes[:g.size, 2] == 0
        values = g._densification_criterium[:g.size][leaf].detach().cpu()
        visible = self.seen[:g.size].cpu()[leaf]
        if not torch.isfinite(values).all():
            raise FloatingPointError('Non-finite detail scores in the split window')
        positive = values[values > 0]
        sample = positive[::max(1, math.ceil(len(positive)/65536))]
        quantiles = torch.quantile(sample, torch.tensor([.5, .9, .99])).tolist() if len(sample) else [0.,0.,0.]
        return dict(iteration=iteration, total_nodes=g.size, leaf_nodes=int(leaf.sum()),
            window_views=self.views, visible_leaves=int(visible.sum()),
            positive_gradient_leaves=int((values > 0).sum()), eligible_leaves=int((values > threshold).sum()),
            threshold=float(threshold), score_max=float(values.max()) if len(values) else 0.,
            score_p50_p90_p99=quantiles, quantiles_sampled=len(sample) != len(positive))

    def reset(self):
        self.seen.zero_()
        self.views = 0
