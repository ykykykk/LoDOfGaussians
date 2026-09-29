"""Build an unpacked Windows runtime; run using the project's .venv Python."""
import json
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dist' / 'YK-Gaussian'


def copy_runtime(source, target):
    result = subprocess.run(['robocopy', str(Path(source).resolve()), str(target),
                             '/E', '/MT:16', '/R:1', '/W:1', '/NFL', '/NDL', '/NJH', '/NJS', '/NP',
                             '/XD', '__pycache__', '.git', '/XF', '*.pyc', '_virtualenv*'])
    if result.returncode >= 8:
        raise RuntimeError(f'Runtime copy failed: {result.returncode}')


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-from', type=Path, help='Reuse an unchanged previous portable Python runtime')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    if OUT.exists():
        raise SystemExit(f'Output already exists; preserve or move it before rebuilding: {OUT}')
    if args.runtime_from and not (args.runtime_from / 'python.exe').is_file():
        raise ValueError('Invalid previous runtime')
    OUT.mkdir(parents=True)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '_virtualenv*')
    print('Copying Python runtime and installed dependencies...', flush=True)
    if args.runtime_from:
        copy_runtime(args.runtime_from, OUT / 'runtime')
    else:
        copy_runtime(sys.base_prefix, OUT / 'runtime')
        copy_runtime(ROOT / '.venv/Lib/site-packages', OUT / 'runtime/Lib/site-packages')
    app = OUT / 'app'
    # Include Qt even when reusing a previous CLI/Tk runtime.
    site = ROOT / '.venv/Lib/site-packages'
    for pattern in ('PySide6', 'shiboken6', 'pyside6_essentials-*.dist-info', 'shiboken6-*.dist-info'):
        matches = list(site.glob(pattern))
        if not matches:
            raise RuntimeError('Install requirements-ui.txt in the build environment first')
        for package in matches:
            shutil.copytree(package, OUT / 'runtime/Lib/site-packages' / package.name,
                            dirs_exist_ok=True, ignore=ignore)
    app.mkdir()
    for name in ['arguments', 'configs', 'csrc', 'gaussian_renderer', 'lpipsPyTorch', 'scene', 'tools', 'utils', 'Docs']:
        shutil.copytree(ROOT / name, app / name, ignore=ignore)
    for path in ROOT.glob('*.py'):
        shutil.copy2(path, app / path.name)
    for name in ['LICENSE.md', 'README.md', 'requirements.txt', 'requirements-ui.txt']:
        shutil.copy2(ROOT / name, app / name)
    hierarchy = Path('submodules/gaussianhierarchy/build/Release')
    shutil.copytree(ROOT / hierarchy, app / hierarchy, ignore=ignore)
    vswhere = Path('C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe')
    vsroot = Path(subprocess.check_output([str(vswhere), '-latest', '-products', '*',
                                          '-property', 'installationPath'], text=True).strip())
    crt_dirs = sorted((vsroot / 'VC/Redist/MSVC').glob('14.*/x64/Microsoft.VC*.CRT'))
    for dll in crt_dirs[-1].glob('*.dll'):
        shutil.copy2(dll, OUT / 'runtime' / dll.name)
        shutil.copy2(dll, app / hierarchy / dll.name)
    for path in (ROOT / 'submodules').rglob('*'):
        if path.is_file() and path.name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE')):
            target = app / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    import torch
    from torch.utils.cpp_extension import get_default_build_root
    cache = Path(get_default_build_root()) / f'py{sys.version_info.major}{sys.version_info.minor}_cu{torch.version.cuda.replace(".", "")}'
    native = OUT / 'native'
    native.mkdir()
    for name in ['gsplat_cuda', 'alod_resident_ops_v2']:
        shutil.copy2(cache / name / (name + '.pyd'), native)
    (app / 'portable_boot.py').write_text(
        "import sys, runpy\nfrom pathlib import Path\n"
        "root = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(root))\n"
        "runpy.run_path(str(root / 'yk_gaussian.py'), run_name='__main__')\n", encoding='utf-8')
    compiler = Path('C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe')
    subprocess.run([str(compiler), '/nologo', '/target:exe', '/platform:x64',
                    '/out:' + str(OUT / 'YK-Gaussian.exe'), str(ROOT / 'scripts/portable_launcher.cs')], check=True)
    (OUT / 'README.txt').write_text(
        'YK Gaussian Studio 0.4.1 portable (Windows x64)\n\n'
        'UI language follows the Windows display language: Chinese or English.\n'
        'Fine model is the model-preparation stage name.\n\n'
        'Double-click YK-Gaussian.exe for the desktop UI. No terminal is needed.\n'
        'Create or open a .ykproject.json project, then follow steps 01 through 07.\n'
        'Each step has basic and advanced parameters; project settings and results are saved.\n'
        'Changing earlier settings marks downstream results stale without deleting files.\n'
        'The interactive 3D viewport is embedded in the Qt desktop workspace.\n'
        'Left drag: orbit; middle/Shift drag: pan; wheel: zoom; F: frame scene.\n'
        'Scene, properties and console panels can be docked or floated.\n'
        'Closing the desktop UI asks before stopping a running task.\n'
        'Use a different preview port if another viewer is already running.\n\n'
        'Keep this entire folder together. Python and CUDA libraries are bundled.\n'
        'A compatible NVIDIA GPU/driver is required. No Python, CUDA Toolkit or compiler installation is needed.\n'
        'Precompiled CUDA extensions target the build machine GPU; run doctor on the target computer.\n\n'
        'YK-Gaussian.exe --help\nYK-Gaussian.exe ui\nYK-Gaussian.exe view --help\n'
        'YK-Gaussian.exe doctor\nYK-Gaussian.exe train --help\n'
        'YK-Gaussian.exe train-paged --help\nYK-Gaussian.exe flat --help\n'
        'YK-Gaussian.exe convert-blocks --help\nYK-Gaussian.exe export-ply --help\n'
        'YK-Gaussian.exe evaluate --help\n\n'
        'Relative paths resolve from your terminal working directory. Bundled configs are in app/configs.\n'
        'See app/README.md and app/LICENSE.md.\n', encoding='utf-8')
    (OUT / 'build-info.json').write_text(json.dumps({
        'version': '0.4.1', 'built_at': datetime.now(timezone.utc).isoformat(),
        'ui_languages': ['zh', 'en'],
        'ui_language_selection': 'Windows display language: Chinese -> zh; otherwise en',
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'working_tree_changes': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        'source_sha256': {str(p.relative_to(app)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in app.rglob('*') if p.suffix in ('.py', '.html')},
        'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
    }, indent=2), encoding='utf-8')
    print(f'Built: {OUT}', flush=True)


if __name__ == '__main__':
    main()
