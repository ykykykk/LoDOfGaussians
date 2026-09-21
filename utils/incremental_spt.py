"""Exact dirty-subtree SPT rebuilds for the CPU-backed resident trainer.

Ranges depend on ALL descendant positions/scales, not just newly split nodes.
An exact block scan catches geometry edits and nonlocal relocation/reparenting.
Clean subtrees keep their SPTs. Packed GPU arrays are patched in place when the
layout is unchanged; a topology/layout change may require a linear repack.
"""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Subtree:
    ids: torch.Tensor
    minimum: torch.Tensor
    maximum: torch.Tensor
    radius: float


def min_distance(properties, nodes, ids, granularity):
    scales = properties[ids, 3:6].exp()
    result = (scales[:, 0] * scales[:, 1] + scales[:, 0] * scales[:, 2]
              + scales[:, 1] * scales[:, 2]).sqrt() / granularity
    # Preserve the upstream singleton/vector leaf sentinel convention.
    result[nodes[ids, 2] == 0] = -1e6 if len(ids) == 1 else -1e9
    return result


def children(nodes, ids, size):
    first = nodes[ids, 3].long()
    if ((first < 0) | (first >= size)).any():
        raise ValueError("invalid first-child index in hierarchy")
    second = nodes[first, 4].long()
    if ((second < 0) | (second >= size)).any() or (first == second).any():
        raise ValueError("invalid binary sibling relation in hierarchy")
    return torch.cat((first, second))


class IncrementalSPT:
    def __init__(self, root_volume, granularity, min_size=256,
                 bounding_spheres=False, device="cuda", scan_rows=65536):
        if root_volume <= 0 or granularity <= 0 or min_size < 0 or scan_rows <= 0:
            raise ValueError("invalid SPT construction options")
        self.root_volume, self.granularity = root_volume, granularity
        self.min_size, self.bounding_spheres = min_size, bounding_spheres
        self.device, self.scan_rows = torch.device(device), int(scan_rows)
        self.cache, self.snapshots = {}, []
        self.owners = np.empty(0, dtype=np.int32)
        self.layout = None
        self.generation = 0
        self.stats = {}

    def _changed_roots(self, g):
        invalid = set()
        size = g.size
        if size < len(self.owners):
            # Compaction changes global IDs; cached entries no longer identify
            # the same Gaussians. A full refresh is required, not a heuristic.
            self.cache.clear()
            self.snapshots.clear()
            self.layout = None
            self.owners = np.empty(0, dtype=np.int32)
        old_size = len(self.owners)
        if size > old_size:
            self.owners = np.pad(self.owners, (0, size - old_size), constant_values=-1)
        for block, start in enumerate(range(0, size, self.scan_rows)):
            stop = min(size, start + self.scan_rows)
            geom = g.properties[start:stop, :6]
            topo = g.nodes[start:stop, :5]
            if block < len(self.snapshots):
                old_geom, old_topo = self.snapshots[block]
                n = min(len(old_geom), len(geom))
                changed = ((geom[:n] != old_geom[:n]).any(1)
                           | (topo[:n] != old_topo[:n]).any(1)).numpy()
                invalid.update(self.owners[start:start + n][changed].tolist())
                if len(geom) == len(old_geom):
                    old_geom.copy_(geom)
                    old_topo.copy_(topo)
                else:
                    self.snapshots[block] = (geom.clone(), topo.clone())
            else:
                self.snapshots.append((geom.clone(), topo.clone()))
        invalid.discard(-1)
        return invalid

    def _partition(self, g):
        nodes = g.nodes
        stack = torch.tensor([g.skybox_points], dtype=torch.long)
        upper, cuts, visited = [], [], 0
        while len(stack):
            visited += len(stack)
            if visited > g.size:
                raise ValueError("hierarchy contains a cycle or repeated child")
            upper.append(stack)
            leaf = nodes[stack, 2] == 0
            cuts.append(stack[leaf])
            internal = stack[~leaf]
            if len(internal):
                descend = g.properties[internal, 3:6].exp().prod(1) > self.root_volume
                cuts.append(internal[~descend])
                stack = children(nodes, internal[descend], g.size)
            else:
                break
        return torch.cat(upper), torch.cat(cuts)

    def _build(self, g, root):
        nodes, props = g.nodes, g.properties
        stack = torch.tensor([root], dtype=torch.long)
        root_min = min_distance(props, nodes, stack, self.granularity)
        ids, mins, maxs = [stack], [root_min], [torch.full((1,), 1e12)]
        bound = float(props[root, 3:6].exp().max()) * 3.0
        inherited = root_min
        count = 1
        while len(stack):
            internal = nodes[stack, 2] > 0
            parents = stack[internal]
            if not len(parents):
                break
            stack = children(nodes, parents, g.size)
            parent_max = inherited[internal].repeat(2)
            count += len(stack)
            if count > g.size:
                raise ValueError("cycle or repeated child in SPT subtree")
            offset = (props[stack, :3] - props[root, :3]).square().sum(1).sqrt()
            bound = max(bound, float((offset + props[stack, 3:6].exp().max(1).values * 3).max()))
            distance = min_distance(props, nodes, stack, self.granularity) + offset
            inherited = torch.minimum(distance, parent_max)
            ids.append(stack)
            mins.append(inherited)
            maxs.append(parent_max)
        ids, lo, hi = torch.cat(ids), torch.cat(mins), torch.cat(maxs)
        order = torch.argsort(hi, descending=True, stable=True)
        return Subtree(ids[order].to(torch.int32), lo[order], hi[order], bound)

    @torch.no_grad()
    def refresh(self, g):
        if g.properties.device.type != "cpu" or g.nodes.device.type != "cpu":
            raise ValueError("incremental SPT construction requires CPU backing")
        if not (0 <= g.skybox_points < g.size < 2**31):
            raise ValueError("invalid hierarchy root/size for int32 SPT indices")
        dirty = self._changed_roots(g)
        upper, cut = self._partition(g)
        kept, rebuilt, small_members = {}, set(), []
        for root in cut.tolist():
            if g.nodes[root, 2] == 0:
                continue
            entry = self.cache.get(root)
            if entry is None or root in dirty:
                entry = self._build(g, root)
                rebuilt.add(root)
            kept[root] = entry
            self.owners[entry.ids.long().numpy()] = root
            if len(entry.ids) <= self.min_size:
                small_members.append(entry.ids[entry.ids != root].long())
        self.cache = kept
        roots = [r for r, e in kept.items() if len(e.ids) > self.min_size]
        layout = tuple((r, len(kept[r].ids)) for r in roots)
        sizes = [n for _, n in layout]
        starts = torch.tensor([0, *np.cumsum(sizes).tolist()], dtype=torch.int32)
        layout_changed = layout != self.layout or not hasattr(g, "SPT_min")
        changed_segments = [i for i, root in enumerate(roots) if root in rebuilt]
        changed_rows = sum(sizes[i] for i in changed_segments)
        # For widespread changes, a few large copies are cheaper than many
        # tiny synchronous H2D patches. Clean CPU subtrees remain reused.
        repack = layout_changed or len(changed_segments) > 64 or changed_rows > sum(sizes) / 2
        uploaded = 0
        fields = (("SPT_gaussian_indices", "ids", torch.int32),
                  ("SPT_min", "minimum", torch.float32),
                  ("SPT_max", "maximum", torch.float32))
        if repack:
            for attr, field, dtype in fields:
                values = [getattr(kept[r], field) for r in roots]
                tensor = torch.cat(values) if values else torch.empty(0, dtype=dtype)
                setattr(g, attr, tensor.to(self.device))
            g.SPT_starts = starts.to(self.device)
            uploaded = sum(sizes)
        else:
            for i, root in enumerate(roots):
                if root in rebuilt:
                    start, stop = int(starts[i]), int(starts[i + 1])
                    for attr, field, _ in fields:
                        getattr(g, attr)[start:stop].copy_(getattr(kept[root], field))
                    uploaded += stop - start
        self.layout = layout
        if small_members:
            upper = torch.cat([upper, *small_members])
        upper = torch.sort(upper).values
        if len(torch.unique(upper)) != len(upper) or int(upper[0]) != g.skybox_points:
            raise ValueError("invalid upper-tree partition")
        rows = g.nodes[upper, :6].clone()
        old_parent = rows[:, 1].long().clamp(min=0)
        rows[:, 5] = upper.to(rows.dtype)
        rows[:, 1] = torch.searchsorted(upper, old_parent).to(rows.dtype)
        rows[0, 1] = -1
        ordinary_leaf = rows[:, 2] == 0
        rows[:, 3] = torch.where(ordinary_leaf, -1,
                                torch.searchsorted(upper, rows[:, 3].long()).to(rows.dtype))
        siblings = rows[:, 4] > 0
        rows[siblings, 4] = torch.searchsorted(upper, rows[siblings, 4].long()).to(rows.dtype)
        root_locations = torch.searchsorted(upper, torch.tensor(roots, dtype=torch.long))
        rows[root_locations, 2] = 0
        rows[root_locations, 3] = torch.arange(len(roots), dtype=rows.dtype)
        xyz = g.properties[upper, :3].clone()
        scales = g.properties[upper, 3:6].clone()
        minimum = min_distance(g.properties, g.nodes, old_parent, self.granularity).square()
        minimum[0] = 1e12
        # Precompute the reference BFS cut-emission order. The parallel CUDA
        # selector can retain this order without a CPU traversal per view.
        stack, order, levels = torch.tensor([0]), [], []
        seen = 0
        while len(stack):
            seen += len(stack)
            if seen > len(rows):
                raise ValueError("invalid compact upper tree")
            leaf = rows[stack, 2] == 0
            order.extend((stack[leaf], stack[~leaf]))
            levels.append(stack)
            stack = children(rows, stack[~leaf], len(rows))
        if seen != len(rows):
            raise ValueError("upper tree contains unreachable nodes")
        if self.bounding_spheres:
            radii = torch.zeros(len(rows), dtype=torch.float32)
            radii[ordinary_leaf] = scales[ordinary_leaf].exp().max(1).values * 3
            radii[root_locations] = torch.tensor([kept[r].radius for r in roots])
            for level in reversed(levels):
                parent = level[rows[level, 2] > 0]
                if not len(parent):
                    continue
                first = rows[parent, 3].long()
                second = rows[first, 4].long()
                a = radii[first] + (xyz[parent] - xyz[first]).square().sum(1).sqrt()
                b = radii[second] + (xyz[parent] - xyz[second]).square().sum(1).sqrt()
                radii[parent] = torch.maximum(a, b)
            g.bounding_sphere_radii = radii.to(self.device)
        g.upper_tree_nodes = rows.to(self.device)
        g.upper_tree_xyz = xyz.to(self.device)
        g.upper_tree_scaling = scales.to(self.device)
        g.upper_tree_bounds = g.upper_tree_scaling.max(1).values.exp() * 3
        g.min_distance_squared = minimum.to(self.device)
        g.upper_cut_order = torch.cat(order).to(device=self.device, dtype=torch.long)
        self.generation += 1
        self.stats = dict(spt_rebuilt=len(rebuilt), spt_reused=len(kept) - len(rebuilt),
                          spt_scanned_rows=g.size, spt_uploaded_rows=uploaded,
                          spt_layout_repack=int(layout_changed), spt_bulk_upload=int(repack), spt_upper_nodes=len(upper),
                          spt_generation=self.generation)
        return torch.tensor(roots, dtype=torch.long)
