"""Local, same-user CLI control of the running Qt workspace.

All dispatch happens on the GUI event thread, including calls from the embedded
command field. The pipe name is scoped to this installation and OS user.
"""
import argparse
import copy
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def server_name():
    identity = str(Path(__file__).resolve().parent).casefold() + '|' + getpass.getuser()
    return 'yk-gaussian-' + hashlib.sha256(identity.encode()).hexdigest()[:24]


class CommandParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def parser():
    p = CommandParser(prog='YK-Gaussian control', description='Control the running GUI; replies are JSON.')
    commands = p.add_subparsers(dest='action', required=True)
    commands.add_parser('help')
    commands.add_parser('status')
    commands.add_parser('run-all')
    commands.add_parser('save', help='Save project and request a training checkpoint if running.')
    creating = commands.add_parser('new')
    creating.add_argument('--project', required=True)
    creating.add_argument('--source', default='')
    creating.add_argument('--output-root', default='')
    opening = commands.add_parser('open')
    opening.add_argument('--project', required=True)
    run = commands.add_parser('run')
    run.add_argument('--step', required=True, choices=('import', 'check', 'initial', 'prepare', 'train', 'evaluate', 'export'))
    commands.add_parser('stop', help='Stop the current task; unsaved iterations are lost.')
    preview = commands.add_parser('preview')
    preview.add_argument('enabled', choices=('on', 'off'))
    setting = commands.add_parser('set')
    setting.add_argument('--step', required=True)
    setting.add_argument('--key', required=True, help='Existing dotted parameter path')
    setting.add_argument('--value', required=True, help='JSON value, e.g. 1500, true, or a quoted string')
    return p


def status(window):
    project = window.project
    process = window.process
    return {'ok': True, 'pid': os.getpid(), 'project': copy.deepcopy(project),
            'selected_step': window.step, 'active_step': window.active_step,
            'task_pid': int(process.processId()) if process else None,
            'busy': bool(process), 'preview_enabled': window.preview_enabled.isChecked(),
            'preview_connected': bool(window.url)}


def dispatch(window, argv):
    """Execute an argv list on the GUI thread; return a JSON-serializable reply."""
    from workflow_project import STEPS, load_project, save_project, update_settings, can_run, create_project
    try:
        if not argv or argv in (['help'], ['--help'], ['-h']):
            return {'ok': True, 'help': parser().format_help(), 'examples': [
                'status', 'open --project "F:/Projects/scene.ykproject.json"',
                'run --step check', 'preview off',
                'set --step initial --key config.iterations --value 1500', 'stop']}
        args = parser().parse_args(argv)
        if args.action == 'status':
            return status(window)
        if args.action == 'save':
            accepted = window.save_project_and_training()
            return dict(status(window), ok=accepted, save_pending=bool(window.save_request_id))
        if args.action == 'preview':
            window.preview_enabled.setChecked(args.enabled == 'on')
            return status(window)
        if args.action == 'stop':
            window.stop(confirmed=True)
            return status(window)
        if window.process:
            raise ValueError('A task is running. Wait for it to finish or use control stop.')
        # Save GUI edits without opening a modal dialog for a CLI client.
        if window.project:
            update_settings(window.project, window.step, window.collect())
            save_project(window.project)
        if args.action == 'run-all':
            if not window.project:
                raise ValueError('Open a project first.')
            window.run_all()
            return dict(status(window), auto_running=window.auto_running)
        if args.action in ('open', 'new'):
            path = Path(args.project).resolve()
            if args.action == 'new':
                if path.exists():
                    raise ValueError('Project already exists; use open.')
                project = create_project(path, source_path=args.source, output_root=args.output_root)
            else:
                project = load_project(str(path))
            window.project = project
            window.step = next((s for s in STEPS if project['steps'][s]['status'] != 'completed'), 'evaluate')
            window.clear_scene()
        else:
            if not window.project:
                raise ValueError('Open a project first with control open --project PATH.')
            if args.step not in STEPS:
                raise ValueError('Unknown workflow step: ' + args.step)
            ready, reason = can_run(window.project, args.step)
            if args.action == 'run' and not ready:
                raise ValueError(reason)
            if args.action == 'set':
                settings = copy.deepcopy(window.project['settings'][args.step])
                keys = args.key.split('.')
                node = settings
                for key in keys[:-1]:
                    node = node[key]
                old = node[keys[-1]]
                value = json.loads(args.value)
                if isinstance(old, bool):
                    valid = isinstance(value, bool)
                elif isinstance(old, float):
                    valid = type(value) in (int, float) and math.isfinite(value)
                    if valid:
                        value = float(value)
                else:
                    valid = type(value) is type(old)
                if not valid or isinstance(old, dict):
                    raise ValueError('Value must match the existing parameter type: ' + type(old).__name__)
                node[keys[-1]] = value
                update_settings(window.project, args.step, settings)
                save_project(window.project)
            window.step = args.step
        window.make_form()
        window.refresh()
        if args.action == 'run':
            window.start()
        return status(window)
    except (Exception, SystemExit) as exc:
        return {'ok': False, 'error': str(exc)}


def install_control(window):
    """Keep server owned by window; never replace another live server."""
    from PySide6.QtNetwork import QLocalServer, QLocalSocket
    from PySide6.QtCore import QTimer
    name = server_name()
    probe = QLocalSocket()
    probe.connectToServer(name)
    if probe.waitForConnected(150):
        probe.disconnectFromServer()
        window.log.appendPlainText('[Control] Another window owns this installation control endpoint.')
        return None
    QLocalServer.removeServer(name)
    server = QLocalServer(window)
    server.setSocketOptions(QLocalServer.UserAccessOption)
    sockets = set()

    def connected():
        while server.hasPendingConnections():
            sock = server.nextPendingConnection()
            sockets.add(sock)
            data = bytearray()
            timer = QTimer(sock)
            timer.setSingleShot(True)
            timer.timeout.connect(sock.abort)
            timer.start(10000)

            def read(sock=sock, data=data, timer=timer):
                data.extend(bytes(sock.readAll()))
                if len(data) > 65536:
                    sock.abort()
                    return
                if b'\n' not in data:
                    return
                timer.stop()
                try:
                    argv = json.loads(bytes(data).split(b'\n', 1)[0])
                    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
                        raise ValueError('Expected an argv string list')
                    response = dispatch(window, argv)
                except Exception as exc:
                    response = {'ok': False, 'error': str(exc)}
                sock.write((json.dumps(response, ensure_ascii=False) + '\n').encode())
                sock.disconnectFromServer()

            def cleanup(sock=sock):
                sockets.discard(sock)
                sock.deleteLater()

            sock.readyRead.connect(read)
            sock.disconnected.connect(cleanup)

    server.newConnection.connect(connected)
    if not server.listen(name):
        window.log.appendPlainText('[Control] ' + server.errorString())
        server.deleteLater()
        return None
    window.control_server = server
    window.log.appendPlainText('[Control] Ready: YK-Gaussian.exe control status')
    return server


def client_main(argv):
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtNetwork import QLocalSocket
    if argv in (['help'], ['--help'], ['-h']) or not argv:
        print(parser().format_help())
        return 0
    try:
        parser().parse_args(argv)
    except ValueError as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}))
        return 2
    app = QCoreApplication.instance() or QCoreApplication(['YK-Gaussian control'])
    sock = QLocalSocket()
    sock.connectToServer(server_name())
    if not sock.waitForConnected(2000):
        print(json.dumps({'ok': False, 'error': 'No running GUI for this installation. Open YK-Gaussian.exe first.'}))
        return 1
    sock.write((json.dumps(argv) + '\n').encode())
    sock.waitForBytesWritten(2000)
    data = bytearray()
    while b'\n' not in data:
        if not sock.bytesAvailable() and not sock.waitForReadyRead(15000):
            print(json.dumps({'ok': False, 'error': 'GUI control reply timed out or disconnected.'}))
            return 1
        data.extend(bytes(sock.readAll()))
    result = json.loads(bytes(data).split(b'\n', 1)[0])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get('ok') else 1
