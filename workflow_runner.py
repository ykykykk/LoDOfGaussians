"""Execute one saved desktop workflow step through the existing training backends."""
import argparse
import copy
import math
import json
import hashlib
import os
from pathlib import Path
import runpy
import sys
import uuid
from datetime import datetime
from workflow_project import STEPS, load_project, save_project, can_run, invalidate, now, DATA_DEPENDENTS, clear_result

ROOT = Path(__file__).resolve().parent

def resume_identity(project, step):
    """Training compatibility, excluding display and checkpoint cadence choices."""
    settings = copy.deepcopy(project['settings'][step])
    for key in ('preview', 'viewer_port', 'resume_latest'):
        settings.pop(key, None)
    for section in ('resident', 'paged'):
        runtime = settings.get('config', {}).get(section, {})
        for key in ('checkpoint_every', 'profile_every'):
            runtime.pop(key, None)
    dependency = 'prepare' if step == 'train' else 'import'
    outputs = copy.deepcopy(project['steps'][dependency]['outputs'])
    paths = {}
    for key, value in outputs.items():
        if key in ('checkpoint', 'report') and value and Path(value).is_file():
            stat = Path(value).stat()
            paths[key] = [str(Path(value).resolve()), stat.st_size, stat.st_mtime_ns]
    value = {'settings': settings, 'source': str(Path(project['settings']['import']['source_path']).resolve()),
             'dependencies': outputs, 'dependency_files': paths}
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
    return digest, value

def resumable_path(previous, signature, step):
    if previous.get('run_signature') != signature or not previous.get('run_dir'):
        return None
    directory = Path(previous['run_dir'])
    if step == 'initial':
        if previous.get('status') == 'completed':
            return None
        candidates = [directory/'resident_latest.pt', directory/'scaffold/scaffold_latest.pt']
        if any(path.is_file() for path in candidates):
            return directory
        if any((directory/'scaffold/point_cloud').glob('iteration_*/point_cloud.ply')):
            return directory
    elif step == 'train' and (directory/'blocks/manifest.json').is_file():
        return directory/'blocks/manifest.json'
    elif step == 'train' and previous.get('resume_checkpoint') and Path(previous['resume_checkpoint']).is_file():
        return Path(previous['resume_checkpoint'])
    return None

def run_script(script, args):
    old = sys.argv
    try:
        sys.argv = [str(ROOT/script)] + [str(x) for x in args]
        try:
            runpy.run_path(str(ROOT/script), run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise RuntimeError(f'{script} exited with {exc.code}') from exc
    finally:
        sys.argv = old

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return str(path)

def checkpoint_info(path):
    path = Path(path).resolve()
    if path.is_dir():
        path = path/'manifest.json'
    if path.suffix.lower() == '.json':
        from utils.gaussian_block_store import GaussianBlockStore
        store = GaussianBlockStore.open(path)
        return str(path), store.metadata, 'paged'
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if not all(k in state for k in ('contract', 'properties', 'nodes', 'size', 'iteration')):
        raise ValueError('Not a supported resident checkpoint')
    return str(path), state, 'resident'

def seed_checkpoint_settings(project, state):
    """Inherit a model's actual contract without overwriting explicit user edits."""
    options = copy.deepcopy(state['contract']['options'])
    if not project['settings']['import'].get('checkpoint'):
        planned = project['settings']['train']['config']['options']
        for key in ('iterations', 'densify_until_iter', 'position_lr_max_steps'):
            if key in planned:
                options[key] = planned[key]
    if state.get('block_rows'):
        layout = {'block_size': int(state['block_rows']), 'radius_bands': bool(state.get('radius_bands', True))}
        edited_layout = project.get('edited_fields', {}).get('prepare', [])
        for key, value in layout.items():
            if key not in edited_layout:
                project['settings']['prepare'][key] = value
    for step, values in (
        ('train', {'config': {'options': options, 'paged': state.get('paged', {})}}),
        ('prepare', {'options': {k: options[k] for k in ('cap_max', 'densify_max_new_nodes', 'iterations', 'densification_interval', 'densify_until_iter') if k in options}})):
        edited = project.get('edited_fields', {}).get(step, [])
        def merge(target, source, prefix=''):
            for key, value in source.items():
                path = prefix+key
                if isinstance(value, dict):
                    merge(target.setdefault(key, {}), value, path+'.')
                elif not any(path == x or path.startswith(x+'.') for x in edited):
                    target[key] = copy.deepcopy(value)
        merge(project['settings'][step], values)


def execute(project, step, run_dir, previous):
    settings = project['settings'][step]
    imported = project['settings']['import']
    source = str(Path(imported['source_path']).resolve()) if imported['source_path'] else ''
    outputs = lambda key: project['steps'][key]['outputs']
    if step == 'import':
        if not source or not Path(source).is_dir():
            raise ValueError('Select an existing COLMAP dataset directory')
        checkpoint = imported.get('checkpoint', '')
        if checkpoint:
            checkpoint, state, kind = checkpoint_info(checkpoint)
            if Path(state['contract']['source']).resolve() != Path(source):
                raise ValueError('Imported checkpoint belongs to another dataset path')
        write_json(run_dir/'import.json', {'source_path': source, 'checkpoint': checkpoint})
        return {'report': str(run_dir/'import.json'), 'source_path': source, 'checkpoint': checkpoint}
    if step == 'check':
        from utils.dataset_preflight import inspect_dataset
        report = inspect_dataset(source, resolution=int(settings['resolution']), hold=int(settings['llff_hold']))
        return {'report': write_json(run_dir/'dataset_check.json', report)}
    if step == 'initial':
        checkpoint = outputs('import').get('checkpoint')
        if checkpoint:
            checkpoint, state, kind = checkpoint_info(checkpoint)
            return {'checkpoint': checkpoint, 'reused': True, 'iteration': int(state['iteration'])}
        config = copy.deepcopy(settings['config'])
        config['viewer'] = {'enabled': True, 'port': 0} if os.environ.get('YK_DESKTOP_PREVIEW') == '1' else {'enabled': settings.get('preview', True), 'port': int(settings.get('viewer_port', 0))}
        # Dataset check already ran; disable legacy auto-policy that overwrites cache budgets.
        config.setdefault('general_policy', {})['enabled'] = False
        every = int(config.get('resident', {}).get('checkpoint_every', 0))
        if not 0 < every < int(config['iterations']):
            raise ValueError('Initial checkpoint_every must be positive and less than iterations')
        if int(config['coarse_iterations']) <= 0:
            raise ValueError('coarse_iterations must be positive')
        config_path = write_json(run_dir/'config.json', config)
        args = ['--project_dir', source, '--output_dir', run_dir, '--config', config_path, '--resolution', settings['resolution'], '--seed', settings['seed'], '--training_backend', 'resident', '--resident_version', '2']
        checkpoint = run_dir/'resident_latest.pt'
        if checkpoint.is_file():
            _, saved, _ = checkpoint_info(checkpoint)
            if int(saved['iteration']) >= int(config['iterations']):
                return {'checkpoint': str(checkpoint), 'config': config_path}
            args.extend(['--resume_checkpoint', checkpoint, '--skip_if_exists'])
        elif any((run_dir/'scaffold/point_cloud').glob('iteration_*/point_cloud.ply')):
            args.append('--skip_if_exists')
        run_script('train.py', args)
        checkpoint = run_dir/'resident_latest.pt'
        checkpoint_info(checkpoint)
        return {'checkpoint': str(checkpoint), 'config': config_path}
    if step == 'prepare':
        checkpoint, state, kind = checkpoint_info(outputs('initial')['checkpoint'])
        if kind == 'paged':
            if int(settings['block_size']) != int(state['block_rows']) or bool(settings['radius_bands']) != bool(state.get('radius_bands', True)):
                raise ValueError('Existing paged layout must retain its block_size and radius_bands; repartitioning is not supported')
            from train_paged import fork_store
            store = fork_store(checkpoint, run_dir/'blocks')
            overrides = settings.get('options', {})
            allowed = {'cap_max', 'densify_max_new_nodes', 'iterations', 'densification_interval', 'densify_until_iter'}
            if set(overrides) - allowed:
                raise ValueError('Unsupported flat migration options')
            store.metadata['contract']['options'].update(overrides)
            if int(store.metadata['contract']['options']['cap_max']) < sum(b['count'] for b in store.blocks):
                raise ValueError('Point cap is smaller than the existing model')
            store.checkpoint()
            inherited = {'block_size': store.block_rows, 'radius_bands': state.get('radius_bands', True)}
            print('Existing paged layout inherited: '+json.dumps(inherited), flush=True)
            return {'checkpoint': str(store.root/'manifest.json'), 'inherited_layout': inherited}
        config = {'resident': {'representation': 'flat'}}
        overrides = settings.get('options', {})
        allowed = {'cap_max', 'densify_max_new_nodes', 'iterations', 'densification_interval', 'densify_until_iter'}
        if set(overrides) - allowed:
            raise ValueError('Unsupported flat migration option: '+str(sorted(set(overrides)-allowed)))
        config.update(overrides)
        config_path = write_json(run_dir/'migration.json', config)
        flat = run_dir/'flat.pt'
        if state['contract'].get('representation') == 'flat':
            from tools.flat_checkpoint import atomic_save
            copied = dict(state)
            copied['contract'] = copy.deepcopy(state['contract'])
            copied['contract']['options'].update(overrides)
            if int(copied['contract']['options']['cap_max']) < int(state['size']):
                raise ValueError('Point cap is smaller than the existing model')
            atomic_save(copied, flat)
        else:
            run_script('tools/flat_checkpoint.py', ['migrate', '--input', checkpoint, '--output', flat, '--config', config_path])
        args = ['--input', flat, '--output', run_dir/'blocks', '--block-size', settings['block_size']]
        if not settings['radius_bands']:
            args.append('--no-radius-bands')
        run_script('tools/convert_block_checkpoint.py', args)
        checkpoint = run_dir/'blocks/manifest.json'
        checkpoint_info(checkpoint)
        return {'checkpoint': str(checkpoint), 'flat_checkpoint': str(flat)}
    if step == 'train':
        checkpoint = outputs('prepare')['checkpoint']
        if settings.get('reuse_model'):
            if not imported.get('checkpoint'):
                raise ValueError('Reuse without training requires an explicitly imported checkpoint')
            checkpoint_info(checkpoint)
            return {'checkpoint': checkpoint, 'reused': True, 'note': '已明确选择复用导入模型，未执行训练。'}
        if settings.get('resume_latest') and previous.get('_resume_checkpoint'):
            checkpoint = previous['_resume_checkpoint']
        elif settings.get('resume_latest') and previous.get('resume_valid') and previous.get('parent_checkpoint') == checkpoint:
            checkpoint = previous['outputs'].get('checkpoint') or checkpoint
        config_path = write_json(run_dir/'config.json', settings['config'])
        if int(settings['steps']) < 0:
            raise ValueError('steps cannot be negative; 0 runs the complete schedule')
        args = ['--checkpoint', checkpoint, '--output-dir', run_dir/'blocks', '--source-path', source, '--config', config_path]
        if int(settings['steps']) > 0:
            args.extend(['--steps', settings['steps']])
        if os.environ.get('YK_DESKTOP_PREVIEW') == '1' or settings.get('preview'):
            import socket
            port = 0 if os.environ.get('YK_DESKTOP_PREVIEW') == '1' else int(settings.get('viewer_port', 0))
            if port == 0:
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0))
                    port = sock.getsockname()[1]
            args.extend(['--viewer', '--viewer-port', port])
        _, before, _ = checkpoint_info(checkpoint)
        progress = before.get('image_equivalent_progress')
        if progress is None and not before.get('camera_visits'):
            progress = before['iteration']
        target_iterations = settings['config'].get('options', {}).get('iterations', before['contract']['options']['iterations'])
        if progress is not None and float(progress) >= float(target_iterations):
            if previous.get('_resume_checkpoint') and previous.get('status') != 'completed':
                return {'checkpoint': str(checkpoint), 'config': config_path}
            raise ValueError('Training endpoint reached; increase iterations before continuing')
        run_script('train_paged.py', args)
        checkpoint = run_dir/'blocks/manifest.json'
        _, after, _ = checkpoint_info(checkpoint)
        if int(after['iteration']) <= int(before['iteration']):
            raise RuntimeError('Training produced no optimizer updates')
        return {'checkpoint': str(checkpoint), 'config': config_path}
    checkpoint = outputs('train')['checkpoint']
    if step == 'evaluate':
        report = run_dir/'quality.json'
        run_script('tools/evaluate_block_quality.py', ['--checkpoint', checkpoint, '--output-json', report, '--camera-limit', settings['camera_limit'], '--tile-size', settings['tile_size'], '--halo', settings['halo'], '--pool-gib', settings['pool_gib'], '--preview-dir', run_dir/'previews'])
        if not report.is_file():
            raise RuntimeError('Evaluation did not produce a report')
        return {'report': str(report), 'checkpoint': checkpoint}
    filename = settings['filename']
    if Path(filename).name != filename or not filename.lower().endswith('.ply'):
        raise ValueError('Export filename must be a plain .ply filename')
    target = run_dir/filename
    run_script('tools/export_block_ply.py', ['--checkpoint', checkpoint, '--output', target])
    if not target.is_file():
        raise RuntimeError('Export did not produce a PLY')
    return {'ply': str(target), 'checkpoint': checkpoint}

def validate_settings(step, settings):
    def finite(value, path='settings'):
        if isinstance(value, dict):
            for key, item in value.items():
                finite(item, path+'.'+key)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
            raise ValueError(path+' must be finite')
    finite(settings)
    positive = {
        'check': ('resolution', 'llff_hold'), 'initial': ('resolution',),
        'prepare': ('block_size',), 'train': (),
        'evaluate': ('camera_limit', 'tile_size', 'pool_gib')}.get(step, ())
    for key in positive:
        if float(settings[key]) <= 0:
            raise ValueError(key+' must be positive')
    config = settings.get('config', {})
    if step == 'train' and int(settings['steps']) < 0:
        raise ValueError('steps cannot be negative')
    options = config.get('options', config)
    for key in ('iterations', 'coarse_iterations', 'cap_max', 'position_lr_max_steps', 'densification_interval'):
        if key in options and float(options[key]) <= 0:
            raise ValueError(key+' must be positive')
    runtime = config.get('paged', config.get('resident', {}))
    for key in ('tile_size', 'tiles_per_camera', 'pool_gib', 'checkpoint_every'):
        if key in runtime and float(runtime[key]) <= 0:
            raise ValueError(key+' must be positive')
    for key in ('headroom_gib', 'image_cache_gib'):
        if key in runtime and float(runtime[key]) < 0:
            raise ValueError(key+' cannot be negative')
    if int(settings.get('halo', runtime.get('halo', 8))) < 5:
        raise ValueError('halo must be at least 5')
    if not 0 <= int(settings.get('viewer_port', 0)) <= 65535:
        raise ValueError('viewer_port must be between 0 and 65535')


def run_step(path, step):
    import copy
    project = load_project(path)
    ready, reason = can_run(project, step)
    if not ready:
        raise ValueError(reason)
    previous = copy.deepcopy(project['steps'][step])
    signature, identity = resume_identity(project, step)
    recovery = resumable_path(previous, signature, step)
    invalidate(project, step, downstream=False)
    state = project['steps'][step]
    import psutil
    state.update(status='running', error='', started=now(), pid=os.getpid(),
                 process_create_time=psutil.Process().create_time(), run_signature=signature,
                 run_identity=identity)
    save_project(project)
    try:
        validate_settings(step, project['settings'][step])
        root = project['settings']['import']['output_root']
        if not root:
            raise ValueError('Select an output directory')
        if step == 'initial' and recovery:
            run_dir = recovery
            print('Resuming saved initial model: '+str(run_dir), flush=True)
        else:
            run_dir = Path(root).resolve()/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+step+'-'+uuid.uuid4().hex[:8])
            run_dir.mkdir(parents=True, exist_ok=False)
        if step == 'train' and recovery and project['settings'][step].get('resume_latest'):
            previous['_resume_checkpoint'] = str(recovery)
            state['resume_checkpoint'] = str(recovery)
            print('Resuming saved fine model: '+str(recovery), flush=True)
        state['run_dir'] = str(run_dir)
        save_project(project)
        result = execute(project, step, run_dir, previous)
        old = previous.get('outputs', {})
        keys = ('source_path', 'checkpoint') if step == 'import' else ('checkpoint',)
        if any(old.get(key) != result.get(key) for key in keys):
            for dependent in DATA_DEPENDENTS[step]:
                clear_result(project, dependent)
        state.update(status='completed', outputs=result, finished=now())
        if step in ('import', 'initial', 'prepare') and result.get('checkpoint'):
            _, metadata, _ = checkpoint_info(result['checkpoint'])
            seed_checkpoint_settings(project, metadata)
        if step == 'train':
            state.update(resume_valid=True, parent_checkpoint=project['steps']['prepare']['outputs']['checkpoint'])
        save_project(project)
        print(json.dumps({'event': 'workflow_completed', 'step': step, 'outputs': result}, ensure_ascii=False), flush=True)
        return result
    except BaseException as exc:
        state.update(status='failed', error=str(exc), finished=now())
        save_project(project)
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--step', choices=STEPS, required=True)
    args = parser.parse_args()
    run_step(args.project, args.step)

if __name__ == '__main__':
    main()
