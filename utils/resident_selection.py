"""Use ALoD's existing upper-tree traversal and CUDA SPT cut.

Keep terminal coarse nodes, ordinary leaves, SPT cuts and independent skybox
separate. first_child==0 is a VALID SPT ID, not an ordinary leaf sentinel.
"""
import torch


def split_upper_cut(nodes, cut):
    rows = nodes[cut]
    is_spt = (rows[:, 2] == 0) & (rows[:, 3] >= 0)
    return cut[is_spt], rows[~is_spt, 5]


@torch.no_grad()
def select_gaussians(gaussians, camera, opt, distance_multiplier):
    from gaussian_hierarchy._C import get_spt_cut_cuda

    g = gaussians
    nodes = g.upper_tree_nodes
    device = nodes.device
    if opt.use_bounding_spheres:
        bounds = g.bounding_sphere_radii
    else:
        bounds = g.scaling_activation(g.upper_tree_scaling.max(dim=-1).values) * 3.0
    planes = g.extract_frustum_planes(camera.full_proj_transform)

    def visible(indices):
        if not opt.use_frustum_culling:
            return torch.ones_like(indices, dtype=torch.bool)
        # Same four-plane sphere test as upstream, kept on the nodes' device.
        points = g.upper_tree_xyz[indices]
        distances = points @ planes[:, :3].T + planes[:, 3]
        return (distances + bounds[indices, None] >= 0).all(dim=1)

    def detail(indices):
        return g.min_distance_squared[indices] > (
            camera.camera_center - g.upper_tree_xyz[indices]).square().sum(dim=-1) * distance_multiplier

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
    # No fake SPT when the view contains only direct nodes or is empty.
    return torch.cat(parts).to(torch.int32)
