"""Persistent, dependency-aware desktop workflow (stdlib only)."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

STEPS = ('import', 'check', 'initial', 'prepare', 'train', 'evaluate', 'export')
STEP_TITLES = dict(zip(STEPS, ('导入数据', '检查数据', '构建初始模型', '准备精细模型', '训练与细化', '检查与评估', '导出成果')))

class Project(dict):
    """Keep the loaded disk revision outside the serialized project data."""
    disk_revision = None
    disk_path = None

class ProjectChangedError(ValueError):
    pass

def disk_revision(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None

def now():
    return datetime.now(timezone.utc).isoformat()

def default_settings():
    from workflow_presets import load_defaults
    return load_defaults()


def create_project(path, name='', source_path='', output_root='', checkpoint=''):
    path = Path(path).resolve()
    project = Project({'version': 1, 'path': str(path), 'name': name or path.stem.replace('.ykproject', ''), 'created': now(), 'settings': default_settings(), 'steps': {s: {'status': 'pending', 'outputs': {}, 'error': ''} for s in STEPS}})
    project['settings']['import'].update(source_path=source_path, output_root=output_root or str(path.parent/(project['name']+'_output')), checkpoint=checkpoint)
    save_project(project)
    return project

def save_project(project, path=None):
    target = Path(path or project['path']).resolve()
    if isinstance(project, Project) and project.disk_path == str(target):
        if disk_revision(target) != project.disk_revision:
            raise ProjectChangedError('Project changed on disk. Newer training progress has been preserved; reload the project before editing.')
    project['path'] = str(target)
    project['updated'] = now()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=target.name+'.', suffix='.tmp', dir=target.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(project, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, target)
        if isinstance(project, Project):
            project.disk_path = str(target)
            project.disk_revision = disk_revision(target)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)

def load_project(path):
    raw = Path(path).read_bytes()
    project = Project(json.loads(raw.decode('utf-8-sig')))
    project.disk_path = str(Path(path).resolve())
    project.disk_revision = hashlib.sha256(raw).hexdigest()
    if project.get('version') != 1:
        raise ValueError('Unsupported project version')
    project['path'] = str(Path(path).resolve())
    defaults = default_settings()['initial']
    for key in ('preview', 'viewer_port'):
        project['settings']['initial'].setdefault(key, defaults[key])
    if project['settings']['train'].get('reuse_model'):
        project['settings']['train']['resume_latest'] = False
    for step in STEPS:
        if step not in project['settings'] or step not in project['steps'] or not isinstance(project['settings'][step], dict):
            raise ValueError('Incomplete workflow project')
        state = project['steps'][step]
        if state.get('status') not in ('pending', 'stale', 'running', 'completed', 'failed', 'interrupted') or not isinstance(state.get('outputs'), dict):
            raise ValueError('Invalid workflow step state')
        if state['status'] == 'running' and state.get('pid'):
            import psutil
            try:
                process = psutil.Process(state['pid'])
                alive = abs(process.create_time() - state.get('process_create_time', 0)) < 0.01
            except psutil.NoSuchProcess:
                alive = False
            except psutil.AccessDenied:
                alive = True
            if not alive:
                state['status'] = 'interrupted'
                state['error'] = 'Training process exited; restart this step to resume its latest compatible checkpoint.'
        if state['status'] == 'completed':
            paths = [value for key, value in state['outputs'].items() if key in ('report', 'checkpoint', 'ply', 'flat_checkpoint', 'config') and value]
            if not paths or any(not Path(value).is_file() for value in paths):
                invalidate(project, step)
    return project

DATA_DEPENDENTS = {
    'import': ('check', 'initial', 'prepare', 'train', 'evaluate', 'export'),
    'check': (), 'initial': ('prepare', 'train', 'evaluate', 'export'),
    'prepare': ('train', 'evaluate', 'export'), 'train': ('evaluate', 'export'),
    'evaluate': (), 'export': ()}

def clear_result(project, step):
    state = project['steps'][step]
    state.update(status='pending', outputs={}, error='')
    for key in ('started', 'finished', 'run_dir', 'resume_valid', 'parent_checkpoint',
                'pid', 'process_create_time', 'run_signature', 'run_identity', 'resume_checkpoint'):
        state.pop(key, None)

def invalidate(project, step, downstream=True):
    for key in (step,) + (DATA_DEPENDENTS[step] if downstream else ()):
        clear_result(project, key)

def result_available(project, step):
    state = project['steps'][step]
    if state['status'] != 'completed':
        return False
    outputs = state['outputs']
    required = 'report' if step in ('import', 'check', 'evaluate') else 'ply' if step == 'export' else 'checkpoint'
    if not outputs.get(required) or not Path(outputs[required]).is_file():
        return False
    return all(Path(value).is_file() for key, value in outputs.items()
               if key in ('report', 'checkpoint', 'ply', 'flat_checkpoint', 'config') and value)

def update_settings(project, step, settings):
    if step == 'train':
        settings = copy.deepcopy(settings)
        if settings.get('reuse_model') and settings.get('resume_latest'):
            if not project['settings'][step].get('resume_latest'):
                settings['reuse_model'] = False
            else:
                settings['resume_latest'] = False
    if settings == project['settings'][step]:
        return False
    if any(x['status'] == 'running' for x in project['steps'].values()):
        raise ValueError('Cannot change settings while a workflow step is running')
    def changes(old, new, prefix=''):
        found = []
        for key in set(old) | set(new):
            path = prefix + key
            if isinstance(old.get(key), dict) and isinstance(new.get(key), dict):
                found.extend(changes(old[key], new[key], path+'.'))
            elif old.get(key) != new.get(key):
                found.append(path)
        return found
    edited = project.setdefault('edited_fields', {}).setdefault(step, [])
    edited[:] = sorted(set(edited) | set(changes(project['settings'][step], settings)))
    changed = changes(project['settings'][step], settings)
    project['settings'][step] = copy.deepcopy(settings)
    operational = {'preview', 'viewer_port', 'output_root', 'resume_latest',
                   'config.resident.checkpoint_every', 'config.paged.checkpoint_every',
                   'config.resident.profile_every', 'config.paged.profile_every'}
    material = [key for key in changed if key not in operational]
    if step == 'initial' and project['settings']['import'].get('checkpoint'):
        material = []
    if material:
        invalidate(project, step)
    return True

def can_run(project, step):
    if step not in STEPS:
        return False, 'Unknown workflow step'
    if any(x['status'] == 'running' for x in project['steps'].values()):
        return False, '已有步骤正在运行'
    for required in STEPS[:STEPS.index(step)]:
        if not result_available(project, required):
            return False, '请先完成：'+STEP_TITLES[required]
    return True, ''

def current_checkpoint(project):
    for step in ('train', 'prepare', 'initial', 'import'):
        state = project['steps'][step]
        checkpoint = state.get('outputs', {}).get('checkpoint')
        if state['status'] == 'completed' and checkpoint and Path(checkpoint).exists():
            return checkpoint
    return ''
