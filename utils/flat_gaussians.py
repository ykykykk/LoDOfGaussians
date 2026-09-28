"""Parent-free Gaussian storage, splitting, and exact finest-point selection."""
import math
import copy
import torch


@torch.no_grad()
def compact_leaf_state(state):
    """Copy a resident checkpoint's background and leaves, including Adam rows."""
    n, sky = int(state['size']), int(state['skybox_points'])
    if not 0 <= sky <= n or state['properties'].shape != (n, 69):
        raise ValueError('Invalid resident checkpoint shape or background prefix')
    ids = torch.cat((torch.arange(sky),
                     torch.nonzero(state['nodes'][sky:n, 2] == 0).flatten() + sky))
    result = dict(state)
    if 'contract' in state:
        result['contract'] = copy.deepcopy(state['contract'])
    result['size'] = len(ids)
    result['properties'] = torch.empty((len(ids), 69), dtype=state['properties'].dtype)
    for start in range(0, len(ids), 65536):
        result['properties'][start:start + 65536].copy_(state['properties'][ids[start:start + 65536]])
    result['nodes'] = torch.zeros((len(ids), 6), dtype=torch.int32)
    result['nodes'][:sky, :3] = -99
    result['scores'] = torch.zeros(len(ids), dtype=torch.float32)
    result.update(seen=None, views=0, empty_windows=0, representation='flat')
    return result


def _rotation(q):
    norm = q.norm(dim=1, keepdim=True)
    q = q / norm.clamp_min(torch.finfo(q.dtype).eps)
    q[norm[:, 0] == 0, 0] = 1
    w, x, y, z = q.unbind(1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), 1).reshape(-1, 3, 3)


@torch.no_grad()
def split_flat(g, opt):
    """Replace parents by two children; return NET added rows. Pool must be flushed.

    Backing capacity must already allow the requested growth. A zero per-window
    budget or fraction means unlimited, matching the existing densification API.
    """
    n, sky = int(g.size), int(g.skybox_points)
    if g.properties.ndim != 2 or g.properties.shape[1] != 69:
        raise ValueError('Flat splitting requires SH1 parameters and Adam (69 columns)')
    scores = g._densification_criterium[sky:n]
    if not torch.isfinite(scores).all():
        raise ValueError('Non-finite flat densification score')
    ids = torch.nonzero(scores > opt.densify_grad_threshold).flatten() + sky
    budget = min(len(ids), int(opt.cap_max) - n, len(g.properties) - n)
    limit = int(getattr(opt, 'densify_max_new_nodes', 0))
    if limit > 0:
        budget = min(budget, limit)
    fraction = float(getattr(opt, 'densify_max_leaf_fraction', 0))
    if fraction > 0:
        budget = min(budget, int((n-sky) * fraction))
    if budget <= 0:
        return 0
    if len(ids) > budget:
        ids = ids[torch.topk(g._densification_criterium[ids], budget, sorted=False).indices]
    # Chunk the mutation so large split budgets do not duplicate all Adam rows.
    for start in range(0, budget, 65536):
        chosen = ids[start:start + 65536]
        child = g.properties[chosen, :23].clone()
        scale = child[:, 3:6].exp()
        alpha = child[:, 13].sigmoid()
        # N=2 specialization of gaussianhierarchy/utils.cu relocation formula.
        a = alpha / (1 + (1-alpha).sqrt())
        coefficient = (1 + (1-alpha).sqrt()) / (2-a/math.sqrt(2))
        new_scale = scale * coefficient[:, None]
        axis = scale.argmax(1)
        row = torch.arange(len(chosen))
        distance = (scale[row, axis].square()-new_scale[row, axis].square()).clamp_min(0).sqrt()
        offset = _rotation(child[:, 6:10])[row, :, axis] * distance[:, None]
        child[:, 3:6] = new_scale.log()
        child[:, 13] = torch.logit(a.clamp(.005, 1-torch.finfo(child.dtype).eps))
        target = slice(n+start, n+start+len(chosen))
        g.properties[target, :23] = child
        g.properties[target, :3] += offset
        child[:, :3] -= offset
        g.properties[chosen, :23] = child
        g.properties[chosen, 23:] = 0
        g.properties[target, 23:] = 0
        g.nodes[chosen] = 0
        g.nodes[target] = 0
    g.size = n + budget
    g._densification_criterium[:g.size].zero_()
    return budget


class FlatSelector:
    """Live xyz/radius mirror: update after Adam, refresh after topology barriers.

    No approximate point budget or LoD substitution. The four side planes match
    GaussianModel.extract_frustum_planes; renderer handles depth clipping.
    """
    def __init__(self, g=None, device='cuda', chunk_size=262144, use_frustum_culling=True):
        self.device = torch.device(device)
        self.chunk_size = int(chunk_size)
        self.use_frustum_culling = use_frustum_culling
        self.generation = 0
        self.stats = {}
        self.bounds = None
        if g is not None:
            self.refresh(g)

    @torch.no_grad()
    def refresh(self, g):
        self.bounds = torch.empty((g.size, 4), dtype=torch.float32, device=self.device)
        for start in range(0, g.size, self.chunk_size):
            raw = g.properties[start:min(start+self.chunk_size, g.size), :6].to(self.device)
            self.bounds[start:start+len(raw), :3] = raw[:, :3]
            self.bounds[start:start+len(raw), 3] = raw[:, 3:6].amax(1).exp()*3
        self.skybox_points = int(g.skybox_points)
        self.generation += 1

    @torch.no_grad()
    def update(self, ids, raw):
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        self.bounds[ids, :3] = raw[:, :3]
        self.bounds[ids, 3] = raw[:, 3:6].amax(1).exp()*3

    @torch.no_grad()
    def select(self, g, camera):
        if self.bounds is None or len(self.bounds) != g.size:
            raise RuntimeError('Flat selector must be refreshed after a topology change')
        if not self.use_frustum_culling:
            ids = torch.arange(g.size, dtype=torch.int32, device=self.device)
        else:
            m = camera.full_proj_transform.to(self.device).T
            planes = torch.stack((m[3]+m[0], m[3]-m[0], m[3]+m[1], m[3]-m[1]))
            planes = planes / planes[:, :3].norm(dim=1, keepdim=True).clamp_min(1e-20)
            parts = [torch.arange(self.skybox_points, dtype=torch.int32, device=self.device)]
            for start in range(self.skybox_points, g.size, self.chunk_size):
                b = self.bounds[start:start+self.chunk_size]
                distance = b[:, :3] @ planes[:, :3].T + planes[:, 3]
                visible = (distance + b[:, 3:4] >= 0).all(1)
                parts.append((visible.nonzero().flatten()+start).to(torch.int32))
            ids = torch.cat(parts)
        self.stats = {'flat_selected': len(ids), 'flat_total': g.size, 'flat_generation': self.generation}
        return ids
