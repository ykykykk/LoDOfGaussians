"""Live scaffold/resident previews, polled by the training thread."""
import io
import numpy as np
import torch
from utils.realtime_viewer import RealtimeViewer, orbit_camera, SnapshotRenderer, snapshot_raw


class InitialViewRenderer:
    def __init__(self, gaussians, pipeline, background, pool=None, select=None, before_render=None):
        self.g, self.pipeline, self.background = gaussians, pipeline, background
        self.pool_getter, self.select = pool, select
        self.before_render = before_render
        self.iteration = 0
        xyz = gaussians._xyz if pool is None else gaussians.properties[:gaussians.size, :3]
        start = min(gaussians.skybox_points, len(xyz)-1)
        positions = xyz[start::max(1, (len(xyz)-start)//4096)].detach().cpu().numpy()
        center = np.median(positions, axis=0)
        radius = float(np.quantile(np.linalg.norm(positions-center, axis=1), .8))
        self.home = dict(target=center.tolist(), distance=max(radius*2.5, .01), yaw=0., pitch=0.)

    @property
    def point_count(self):
        return len(self.g._xyz) if self.pool_getter is None else self.g.size

    @torch.no_grad()
    def snapshot(self, limit):
        g = self.g
        count = self.point_count
        stride = 1 if limit <= 0 else max(1, (count + limit - 1)//limit)
        if self.pool_getter is None:
            # Slice raw properties before activation so work/memory is bounded.
            args = (g._xyz[::stride].detach().clone(), g._opacity[::stride].detach().sigmoid(),
                    g._scaling[::stride].detach().exp(),
                    torch.nn.functional.normalize(g._rotation[::stride].detach(), dim=1),
                    g._features_dc[::stride].detach().clone(), g._features_rest[::stride].detach().clone())
            return SnapshotRenderer(args, self.pipeline, self.background.clone(),
                                    g.active_sh_degree, self.home, self.iteration)
        pool = self.pool_getter()
        ids = np.arange(0, count, stride, dtype=np.int64)
        # Hierarchical parents overlap their descendants. Only leaf nodes and
        # the background belong in this approximate training preview.
        if hasattr(g, 'nodes'):
            selected = torch.from_numpy(ids)
            keep = (g.nodes[selected, 2] == 0) | (selected < g.skybox_points)
            ids = ids[keep.cpu().numpy()]
        width = g.properties.shape[1]//3
        raw = g.properties[torch.from_numpy(ids), :width].to(self.background.device)
        # Overlay dirty resident parameters without flushing, replacing the
        # active cut, or resetting the trainer's optimizer graph.
        if pool.state is not None:
            slots = ids if hasattr(pool, 'live_size') and not len(pool.to_slot) else pool.to_slot[ids]
            present = np.flatnonzero(slots >= 0)
            if len(present):
                copy_stream = getattr(pool, '_copy_stream', None)
                if copy_stream is not None:
                    torch.cuda.current_stream(pool.device).wait_stream(copy_stream)
                dst = torch.as_tensor(present, device=pool.device)
                src = torch.as_tensor(slots[present], device=pool.device)
                raw.index_copy_(0, dst, pool.state.index_select(0, src)[:, :width])
        return snapshot_raw(raw, self.pipeline, self.background, g.active_sh_degree, self.home, self.iteration)

    @torch.no_grad()
    def render(self, request):
        from gaussian_renderer import render_gsplat
        from PIL import Image
        camera = orbit_camera(request, self.background.device)
        g = self.g
        if self.pool_getter is None:
            args = (g._xyz, g.get_opacity, g.get_scaling, g.get_rotation,
                    g._features_dc, g._features_rest)
            count = len(g._xyz)
        else:
            if self.before_render is not None:
                self.before_render()
            pool = self.pool_getter()
            ids = self.select(camera, pool.capacity)
            packet = pool.acquire(ids)
            raw = packet.state[:, :packet.width]
            count = len(raw)
            args = (raw[:, :3].contiguous(), raw[:, 13:14].sigmoid(), raw[:, 3:6].exp(),
                    torch.nn.functional.normalize(raw[:, 6:10], dim=1),
                    raw[:, 10:13, None].transpose(1, 2), raw[:, 14:].reshape(count, -1, 3)) if count else None
        if count:
            rgb = render_gsplat(camera, *args, self.pipeline, self.background,
                                sh_degree=g.active_sh_degree)['render']
        else:
            rgb = torch.zeros((3, camera.image_height, camera.image_width), device=self.background.device)
        pixels = (rgb.clamp(0, 1)*255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format='JPEG', quality=88)
        return buffer.getvalue(), count


def start_initial_viewer(config, gaussians, pipeline, background, **kwargs):
    if not config or not config.get('enabled'):
        return None
    viewer = RealtimeViewer(InitialViewRenderer(gaussians, pipeline, background, **kwargs),
                            port=int(config.get('port', 0)))
    print(f'Realtime viewer: {viewer.url}', flush=True)
    return viewer
