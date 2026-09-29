"""Independent camera rendering from bounded immutable training snapshots."""
import io
import json
import math
import threading
from contextlib import nullcontext
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from utils.adam_graph import PREVIEW_CAPTURE_LOCK



def orbit_camera(request, device):
    from utils.graphics_utils import getProjectionMatrix
    width, height = int(request.get('width', 960)), int(request.get('height', 640))
    if not (64 <= width <= 1920 and 64 <= height <= 1080):
        raise ValueError('Preview size must be between 64x64 and 1920x1080')
    target = np.asarray(request['target'], dtype=float)
    distance, yaw, pitch = [float(request[k]) for k in ('distance', 'yaw', 'pitch')]
    if target.shape != (3,) or not np.isfinite(target).all() or not all(map(math.isfinite, (distance, yaw, pitch))) or distance <= 0:
        raise ValueError('Invalid orbit camera')
    pitch = max(-1.55, min(1.55, pitch))
    offset = np.array([math.sin(yaw)*math.cos(pitch), math.sin(pitch), math.cos(yaw)*math.cos(pitch)])
    up_axis = request.get('up_axis', 'Z')
    if up_axis not in ('Y', 'Z'):
        raise ValueError('up_axis must be Y or Z')
    if up_axis == 'Z':
        offset = offset[[0, 2, 1]] * [1, -1, 1]
    eye = target + distance*offset
    forward = -offset
    right = np.cross(forward, [0., 0., 1.] if up_axis == 'Z' else [0., 1., 0.]); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = np.stack((right, down, forward))
    view[:3, 3] = -view[:3, :3] @ eye
    world = torch.tensor(view.T.copy(), device=device)
    fovy = math.radians(55)
    fovx = 2*math.atan(math.tan(fovy/2)*width/height)
    projection = getProjectionMatrix(.001, 1e6, fovx, fovy, .5, .5).T.to(device)
    camera = SimpleNamespace(image_width=width, image_height=height, FoVx=fovx, FoVy=fovy,
        world_view_transform=world, full_proj_transform=world @ projection,
        camera_center=torch.tensor(eye, dtype=torch.float32, device=device),
        focal_length=height/(2*math.tan(fovy/2)))
    return camera


class SnapshotRenderer:
    """Own bounded, immutable CUDA tensors; never access a training pool."""
    def __init__(self, args, pipeline, background, degree, home, iteration):
        self.args, self.pipeline, self.background = args, pipeline, background
        self.degree, self.home, self.iteration = degree, home, iteration
        self.ready = None
        if background.is_cuda:
            self.ready = torch.cuda.Event()
            self.ready.record(torch.cuda.current_stream(background.device))

    @property
    def point_count(self):
        return len(self.args[0])

    @torch.no_grad()
    def render(self, request):
        from gaussian_renderer import render_gsplat
        from PIL import Image
        if self.ready is not None:
            torch.cuda.current_stream(self.background.device).wait_event(self.ready)
        camera = orbit_camera(request, self.background.device)
        rgb = render_gsplat(camera, *self.args, self.pipeline, self.background,
                            sh_degree=self.degree)['render'] if self.point_count else torch.zeros(
                                (3, camera.image_height, camera.image_width), device=self.background.device)
        pixels = (rgb.clamp(0, 1)*255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format='JPEG', quality=85)
        return buffer.getvalue(), self.point_count


def snapshot_raw(raw, pipeline, background, degree, home, iteration):
    count = len(raw)
    args = (raw[:, :3].contiguous(), raw[:, 13:14].sigmoid(), raw[:, 3:6].exp(),
            torch.nn.functional.normalize(raw[:, 6:10], dim=1),
            raw[:, 10:13, None].transpose(1, 2).contiguous(), raw[:, 14:].reshape(count, (raw.shape[1]-14)//3, 3).contiguous())
    return SnapshotRenderer(args, pipeline, background.clone(), degree, home, iteration)


class PagedViewRenderer:
    """Render either a read-only checkpoint pool or the trainer's live pool."""

    def __init__(self, pool, pipeline):
        self.pool, self.pipeline = pool, pipeline
        self.background = torch.zeros(3, device=pool.device)
        bounds = [b for b in pool.store.blocks if not b.get('skybox')]
        if not bounds:
            raise ValueError('Checkpoint contains no scene blocks')
        # Support bounds include large translucent outliers; use a bounded
        # position sample to frame the useful scene instead of those radii.
        samples = []
        for block in bounds:
            rows = pool.store.read(block['id'])
            sample = rows[::max(1, len(rows)//128)]
            samples.append(np.array(sample[sample[:, 13] > -2.2, :3]))
        positions = np.concatenate(samples)
        if not len(positions):
            positions = np.array([b['bounds_min'] for b in bounds] + [b['bounds_max'] for b in bounds])
        center = np.median(positions, axis=0)
        radius = float(np.quantile(np.linalg.norm(positions-center, axis=1), .8))
        self.home = dict(target=center.tolist(), distance=max(radius*2.5, .01), yaw=0., pitch=0.)
        self.iteration = pool.store.metadata.get('iteration', 0)

    @property
    def point_count(self):
        return sum(b['count'] for b in self.pool.store.blocks)

    @torch.no_grad()
    def snapshot(self, limit):
        # Sample global row positions proportionally across blocks, bounded by
        # limit. Read mapped disk rows only for pages not currently resident.
        total = self.point_count
        stride = 1 if limit <= 0 else max(1, math.ceil(total / limit))
        chunks, offset = [], 0
        for block in self.pool.store.blocks:
            count, bid = block['count'], block['id']
            start = (-offset) % stride
            offset += count
            if start >= count:
                continue
            if bid in self.pool.resident:
                chunks.append(self.pool.state[self.pool._slice(bid)][start:count:stride, :23].clone())
            else:
                rows = self.pool.store.read(bid)
                chunks.append(torch.from_numpy(np.array(rows[start:count:stride, :23])).to(self.pool.device))
        raw = torch.cat(chunks) if chunks else torch.empty((0, 23), device=self.pool.device)
        return snapshot_raw(raw, self.pipeline, self.background, 1, self.home, self.iteration)

    @torch.no_grad()
    def render(self, request):
        from gaussian_renderer import render_gsplat
        from utils.graphics_utils import getProjectionMatrix
        from PIL import Image

        camera = orbit_camera(request, self.pool.device)
        width, height = camera.image_width, camera.image_height
        packet = self.pool.acquire(self.pool.candidate_blocks(camera), camera)
        try:
            raw = packet.state
            if len(raw):
                rgb = render_gsplat(camera, raw[:, :3].contiguous(), raw[:, 13:14].sigmoid(),
                    raw[:, 3:6].exp(), torch.nn.functional.normalize(raw[:, 6:10], dim=1),
                    raw[:, 10:13, None].transpose(1, 2), raw[:, 14:].reshape(len(raw), -1, 3),
                    self.pipeline, self.background, sh_degree=1)['render']
            else:
                rgb = torch.zeros((3, height, width), device=self.pool.device)
            pixels = (rgb.clamp(0, 1)*255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format='JPEG', quality=88)
            return buffer.getvalue(), len(raw)
        finally:
            self.pool.finish_step(packet)


class RealtimeViewer:
    """Call poll() between optimizer steps, or serve() for standalone viewing.

    Training publishes bounded snapshots; HTTP rendering owns a separate stream.
    No HTTP worker may read or acquire mutable training pool storage.
    """

    def __init__(self, renderer, port=8765):
        self.renderer = renderer
        self.snapshot = None
        self.enabled = True
        self.last_request = 0.
        self.last_snapshot = 0.
        self.thread = None
        self.stream = None
        self.standalone = False
        self.metrics = dict(renders=0, snapshots=0, render_ms=0., snapshot_ms=0.)
        config_file = Path(__file__).resolve().parents[1] / 'configs' / 'preview.json'
        self.config = json.loads(config_file.read_text(encoding='utf-8')) if config_file.exists() else {}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(1.)

            def log_message(self, *_):
                pass

            def reply(self, code, body, kind, headers=None):
                self.send_response(code)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                for key, value in (headers or {}).items():
                    self.send_header(key, str(value))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass

            def do_GET(self):
                if self.path == '/':
                    self.reply(200, (Path(__file__).parent/'realtime_viewer.html').read_bytes(), 'text/html; charset=utf-8')
                elif self.path == '/api/state':
                    self.reply(200, json.dumps(dict(home=renderer.home, iteration=renderer.iteration,
                        points=renderer.point_count, preview_config=owner.config, preview_enabled=owner.enabled, **owner.metrics)).encode(), 'application/json')
                else:
                    self.reply(404, b'Not found', 'text/plain')

            def do_POST(self):
                if self.path not in ('/api/render', '/api/preview'):
                    self.reply(404, b'Not found', 'text/plain'); return
                # Browser requests must originate from this local viewer.
                origin = self.headers.get('Origin')
                if origin and origin != owner.url:
                    self.reply(403, b'Invalid origin', 'text/plain'); return
                try:
                    size = int(self.headers.get('Content-Length', 0))
                    if not 0 < size <= 8192:
                        raise ValueError('Invalid request size')
                    request = json.loads(self.rfile.read(size))
                    if self.path == '/api/preview':
                        owner.enabled = bool(request.get('enabled', True))
                        if not owner.enabled:
                            owner.snapshot = None
                        self.reply(200, b'{}', 'application/json'); return
                    if not owner.enabled:
                        self.reply(409, b'Preview disabled', 'text/plain'); return
                    owner.last_request = time.monotonic()
                    current = renderer if owner.standalone else owner.snapshot
                    if current is None:
                        self.reply(503, b'Waiting for training snapshot', 'text/plain'); return
                    started = time.perf_counter()
                    with PREVIEW_CAPTURE_LOCK:
                        if not owner.standalone and owner.stream is None and current.background.is_cuda:
                            owner.stream = torch.cuda.Stream(device=current.background.device)
                        with torch.cuda.stream(owner.stream) if owner.stream is not None else nullcontext():
                            body, count = current.render(request)
                    owner.metrics['renders'] += 1
                    owner.metrics['render_ms'] = (time.perf_counter()-started)*1000
                    self.reply(200, body, 'image/jpeg', {'X-Render-Ms': round((time.perf_counter()-started)*1000, 1),
                        'X-Visible-Points': count, 'X-Iteration': current.iteration})
                except (ValueError, KeyError, TypeError) as exc:
                    self.reply(400, str(exc).encode(), 'text/plain; charset=utf-8')
                except RuntimeError as exc:
                    self.reply(503, str(exc).encode(), 'text/plain; charset=utf-8')

        self.server = HTTPServer(('127.0.0.1', port), Handler)
        self.server.timeout = 0
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def poll(self, iteration=None):
        if iteration is not None:
            self.renderer.iteration = iteration
        if self.thread is None:
            self.thread = threading.Thread(target=self.server.serve_forever,
                                           kwargs={'poll_interval': .05}, daemon=True)
            self.thread.start()
        now = time.monotonic()
        if not self.enabled or now-self.last_request > 2.:
            self.snapshot = None
            return
        if now-self.last_snapshot < float(self.config.get('snapshot_interval_s', 1)):
            return
        started = time.perf_counter()
        try:
            snapshot = self.renderer.snapshot(max(0, int(self.config.get('snapshot_points', 0))))
        except (RuntimeError, ValueError, IndexError) as exc:
            self.enabled = False
            self.snapshot = None
            self.metrics['error'] = str(exc)
            print(f'Preview disabled after snapshot error (training continues): {exc}', flush=True)
            return
        if self.enabled:
            self.snapshot = snapshot
        self.last_snapshot = time.monotonic()
        self.metrics['snapshots'] += 1
        self.metrics['snapshot_ms'] = (time.perf_counter()-started)*1000

    def serve(self):
        self.standalone = True
        try:
            self.server.serve_forever(poll_interval=.1)
        finally:
            self.close()

    def close(self):
        if self.thread is not None:
            self.server.shutdown()
            self.thread.join(timeout=2.)
        self.server.server_close()
        self.snapshot = None
