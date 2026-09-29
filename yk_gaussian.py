"""Command line entry point for YK Gaussian."""
import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    'workflow': ('workflow_runner.py', 'Run one saved project workflow step'),
    'view': ('realtime_viewer.py', 'Interactively view a paged checkpoint'),
    'train': ('train.py', 'Prepare and train a scene'),
    'train-paged': ('train_paged.py', 'Train an SSD-paged checkpoint'),
    'flat': ('tools/flat_checkpoint.py', 'Migrate or export a flat checkpoint'),
    'convert-blocks': ('tools/convert_block_checkpoint.py', 'Convert flat checkpoint to disk blocks'),
    'export-ply': ('tools/export_block_ply.py', 'Export a paged checkpoint to PLY'),
    'evaluate': ('tools/evaluate_block_quality.py', 'Evaluate a paged checkpoint'),
}


def portable_backends():
    native = ROOT.parent / 'native'
    if not native.is_dir():
        return
    sys.path.insert(0, str(native))
    import torch
    import gsplat_cuda
    import alod_resident_ops_v2
    # Keep original extension module names (their PyInit exports are fixed).
    sys.modules['gsplat.csrc'] = gsplat_cuda
    import utils.resident_native
    utils.resident_native._MODULE = alod_resident_ops_v2


def main():
    parser = argparse.ArgumentParser(prog='yk-gaussian', description='YK Gaussian portable CLI')
    parser.add_argument('--version', action='version', version='YK Gaussian Studio 0.4.0')
    parser.add_argument('command', nargs='?', choices=[*COMMANDS, 'doctor', 'ui', 'control'],
                        help='; '.join(f'{k}: {v[1]}' for k, v in COMMANDS.items()))
    if len(sys.argv) == 1:
        from desktop_ui import main as ui_main
        ui_main()
        return
    if sys.argv[1] in ('--help', '-h', '--version'):
        parser.parse_args()
        return
    args = parser.parse_args(sys.argv[1:2])
    if args.command == 'ui':
        from desktop_ui import main as ui_main
        ui_main()
        return
    if args.command == 'control':
        from studio_control import client_main
        raise SystemExit(client_main(sys.argv[2:]))
    portable_backends()
    if args.command == 'doctor':
        import json
        import torch
        from gsplat.cuda._backend import _C
        from utils.resident_native import load_native
        import simple_knn._C
        import gaussian_hierarchy._C
        import fused_ssim_cuda
        available = torch.cuda.is_available()
        if available:
            x = torch.ones(4, device='cuda')
            assert (x + x).sum().item() == 8
            native = load_native('cuda')
            mask = native.flat_visible(torch.zeros((1, 4), device='cuda'),
                                       torch.zeros((4, 4), device='cuda'))
            assert mask.numel() == 1
            from fused_ssim import fused_ssim
            pixels = torch.rand((1, 3, 16, 16), device='cuda')
            assert torch.isfinite(fused_ssim(pixels, pixels)).all().item()
            from gsplat import rasterization
            means = torch.tensor([[0., 0., 3.]], device='cuda', requires_grad=True)
            rendered, _, _ = rasterization(
                means, torch.tensor([[1., 0., 0., 0.]], device='cuda'),
                torch.full((1, 3), 0.1, device='cuda'), torch.ones(1, device='cuda') * 0.5,
                torch.ones((1, 3), device='cuda'), torch.eye(4, device='cuda')[None],
                torch.tensor([[[20., 0., 8.], [0., 20., 8.], [0., 0., 1.]]], device='cuda'),
                16, 16)
            rendered.sum().backward()
            assert torch.isfinite(means.grad).all().item()
            torch.cuda.synchronize()
        print(json.dumps({'python': sys.executable, 'torch': torch.__version__,
                         'cuda_available': available,
                         'gpu': torch.cuda.get_device_name() if available else None,
                         'gsplat': _C.__file__,
                         'resident': load_native('cuda').__file__ if available else None}, indent=2))
        if not available:
            raise SystemExit(1)
        return
    script = ROOT / COMMANDS[args.command][0]
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
