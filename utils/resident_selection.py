"""Exact upper-tree selection plus ALoD's existing CUDA SPT cut."""
import torch


@torch.no_grad()
def select_with_budget(g, camera, opt, multiplier, native=None, max_active_nodes=0):
    """Use coarser hierarchy cuts, never truncate arbitrary Gaussian rows."""
    ids = select_gaussians(g, camera, opt, multiplier, native=native)
    if max_active_nodes <= 0 or len(ids) <= max_active_nodes:
        return ids, multiplier
    low, high = multiplier, multiplier
    for _ in range(20):
        high *= 2
        candidate = select_gaussians(g, camera, opt, high, native=native)
        if len(candidate) <= max_active_nodes:
            ids = candidate
            break
        low = high
    else:
        raise RuntimeError('Active-node budget cannot fit even a coarse hierarchy cut')
    # Recover detail between the last over-budget and first fitting cuts.
    for _ in range(4):
        middle = (low + high) / 2
        candidate = select_gaussians(g, camera, opt, middle, native=native)
        if len(candidate) <= max_active_nodes:
            high, ids = middle, candidate
        else:
            low = middle
    return ids, high


def split_upper_cut(nodes, cut):
    rows = nodes[cut]
    is_spt = (rows[:, 2] == 0) & (rows[:, 3] >= 0)
    return cut[is_spt], rows[~is_spt, 5]


@torch.no_grad()
def select_gaussians(gaussians, camera, opt, distance_multiplier, native=None):
    from gaussian_hierarchy._C import get_spt_cut_cuda
    g = gaussians
    nodes = g.upper_tree_nodes
    device = nodes.device
    if opt.use_bounding_spheres:
        bounds = g.bounding_sphere_radii
    else:
        bounds = getattr(g, "upper_tree_bounds", None)
        if bounds is None:
            bounds = g.scaling_activation(g.upper_tree_scaling.max(dim=-1).values) * 3.0
    planes = g.extract_frustum_planes(camera.full_proj_transform)
    weight = distance_multiplier ** 2 if getattr(opt, "lod_pixel_consistent", False) else distance_multiplier
    if native is not None and device.type == "cuda":
        mask = native.upper_cut(nodes.contiguous(), g.upper_tree_xyz.contiguous(), bounds.contiguous(),
                                g.min_distance_squared.contiguous(), planes.contiguous(),
                                camera.camera_center.contiguous(), float(weight),
                                opt.use_frustum_culling)
        order = getattr(g, "upper_cut_order", None)
        if order is None:
            order = torch.arange(len(nodes), device=device)
        cut = order[mask[order]]
    else:
        def visible(indices):
            if not opt.use_frustum_culling:
                return torch.ones_like(indices, dtype=torch.bool)
            points = g.upper_tree_xyz[indices]
            distances = points @ planes[:, :3].T + planes[:, 3]
            return (distances + bounds[indices, None] >= 0).all(dim=1)

        def detail(indices):
            return g.min_distance_squared[indices] > (
                camera.camera_center - g.upper_tree_xyz[indices]).square().sum(dim=-1) * weight

        cut = g.cut_hierarchy_on_condition(nodes, detail, return_upper_tree=False,
                                           root_node=0, leave_out_of_cut_condition=visible)
    spt_nodes, direct = split_upper_cut(nodes, cut)
    parts = [torch.arange(g.skybox_points, device=device, dtype=torch.int32), direct]
    if len(spt_nodes):
        spt_ids = nodes[spt_nodes, 3].contiguous()
        distance = (g.upper_tree_xyz[spt_nodes] - camera.camera_center).square().sum(dim=-1).sqrt()
        distance = (distance * distance_multiplier).contiguous()
        selected, _ = get_spt_cut_cuda(len(spt_ids), g.SPT_gaussian_indices,
                                     g.SPT_starts, g.SPT_max, g.SPT_min, spt_ids, distance)
        parts.append(selected)
    return torch.cat(parts).to(torch.int32)
