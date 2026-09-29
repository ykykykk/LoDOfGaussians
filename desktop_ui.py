"""Qt desktop workspace with a native interactive Gaussian viewport."""
from ui_i18n import tr
import copy
import codecs
import json
import math
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QFont, QImage, QPainter, QPen, QLinearGradient
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QToolBar, QDockWidget, QTreeWidget, QTreeWidgetItem, QComboBox, QLineEdit, QFileDialog, QFormLayout, QScrollArea, QPlainTextEdit, QMessageBox, QFrame, QCheckBox, QProgressBar
from desktop_tasks import TASKS
ROOT = Path(__file__).resolve().parent
STYLE = '\nQWidget { background:#25272c; color:#d6d8de; font-family:"Segoe UI","Microsoft YaHei UI"; font-size:12px; }\nQMainWindow::separator { background:#17191c; width:5px; height:5px; }\nQMenuBar { background:#1c1e22; padding:4px; } QMenuBar::item:selected,QMenu::item:selected { background:#3c4656; }\nQMenu { border:1px solid #454952; padding:5px; }\nQToolBar { background:#2c2f35; border:0; border-bottom:1px solid #181a1e; spacing:7px; padding:7px; }\nQDockWidget { font-weight:600; } QDockWidget::title { background:#30333a; padding:8px; }\nQPushButton { background:#363a43; border:1px solid #484d59; border-radius:4px; padding:7px 12px; }\nQPushButton:hover { background:#464c58; border-color:#727d90; }\nQPushButton:disabled { color:#737780; background:#2b2e34; border-color:#383b42; }\nQPushButton#primary { background:#477acc; border:1px solid #6497e4; color:white; font-weight:600; }\nQPushButton#primary:hover { background:#568cdf; }\nQPushButton#primary:disabled { background:#34445c; border-color:#48556a; color:#8f9bb0; }\nQLineEdit,QComboBox { background:#1e2025; border:1px solid #42464f; border-radius:3px; padding:6px; selection-background-color:#4268a0; }\nQLineEdit:focus,QComboBox:focus { border-color:#6c98dc; }\nQComboBox QAbstractItemView { background:#292c32; selection-background-color:#405c85; }\nQTreeWidget { background:#23252a; border:0; outline:0; } QTreeWidget::item { padding:6px; }\nQTreeWidget::item:selected { background:#344a6b; }\nQPlainTextEdit { background:#1c1e23; border:0; font-family:Consolas; font-size:11px; }\nQStatusBar { background:#1b1d21; color:#a7adba; } QScrollArea { border:0; }\nQTabWidget::pane { border:1px solid #41454e; background:#25272c; }\nQTabBar::tab { background:#2d3037; color:#a9b2c1; border:1px solid #41454e; padding:7px 12px; }\nQTabBar::tab:selected { background:#394458; color:#e1e8f5; border-bottom:2px solid #739ddd; }\nQWidget#parameterPage { background:#25272c; }\nQScrollBar:vertical { background:#22242a; width:9px; } QScrollBar::handle:vertical { background:#515662; min-height:24px; border-radius:4px; }\nQScrollBar::add-line:vertical,QScrollBar::sub-line:vertical { height:0; }\nQProgressBar { border:0; background:#1d2025; height:3px; } QProgressBar::chunk { background:#6c9de7; }\nQLabel#muted { color:#8e96a6; } QLabel#section { color:#aab8ce; font-weight:600; padding-top:12px; padding-bottom:4px; }\n'

class Viewport(QWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.image = QImage()
        self.empty_title = tr('打开一个 Gaussian 场景')
        self.empty_hint = tr('选择分页检查点  ·  打开场景  ·  在此视口中交互')
        self.camera = None
        self.home = None
        self.up_axis = 'Z'
        self.last = None
        self.pan = False
        self.caption = tr('透视视图')
        self.error = ''
        self.setMinimumSize(420, 300)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        bg = QLinearGradient(0, 0, 0, self.height())
        bg.setColorAt(0, QColor('#30343c'))
        bg.setColorAt(1, QColor('#23262c'))
        p.fillRect(self.rect(), bg)
        if not self.image.isNull():
            size = self.image.size().scaled(self.size(), Qt.KeepAspectRatio)
            (x, y) = ((self.width() - size.width()) // 2, (self.height() - size.height()) // 2)
            p.drawImage(x, y, self.image.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            horizon = self.height() * 0.38
            cx = self.width() * 0.5
            p.setPen(QPen(QColor('#40454e'), 1))
            for i in range(-16, 17):
                p.drawLine(int(cx + i * 22), int(horizon), int(cx + i * 100), self.height())
            for frac in [0.08, 0.14, 0.23, 0.35, 0.51, 0.72, 0.96]:
                y = int(horizon + (self.height() - horizon) * frac)
                p.drawLine(0, y, self.width(), y)
            p.setPen(QPen(QColor('#5e4248'), 1))
            p.drawLine(0, int(self.height() * 0.75), self.width(), int(self.height() * 0.75))
            p.setPen(QPen(QColor('#426052'), 1))
            p.drawLine(int(cx), int(horizon), int(cx), self.height())
            p.fillRect(int(cx - 220), int(self.height() / 2 - 72), 440, 128, QColor(30, 33, 39, 225))
            p.setPen(QColor('#e4e8ef'))
            p.setFont(QFont('Segoe UI', 18))
            p.drawText(self.rect().adjusted(0, -25, 0, -25), Qt.AlignCenter, self.empty_title)
            p.setPen(QColor('#9fa8b8'))
            p.setFont(QFont('Segoe UI', 10))
            p.drawText(self.rect().adjusted(0, 38, 0, 38), Qt.AlignCenter, self.empty_hint)
        p.fillRect(9, 9, 220, 25, QColor(24, 27, 32, 190))
        p.fillRect(9, self.height() - 34, min(540, self.width() - 18), 27, QColor(24, 27, 32, 190))
        p.setFont(QFont('Segoe UI', 9))
        p.setPen(QColor('#c2c9d6'))
        p.drawText(16, 25, self.caption)
        p.setPen(QColor('#949eae'))
        p.drawText(16, self.height() - 15, tr('左键旋转   |   中键 / Shift+拖动平移   |   滚轮缩放   |   F 复位'))
        (cx, cy) = (self.width() - 46, 49)
        axes = [(24, 10, '#d98186', 'X'), (0, -25, '#92c88c', 'Y'), (-19, 15, '#80a9e9', 'Z')] if self.up_axis == 'Y' else [(24, 10, '#d98186', 'X'), (-19, 15, '#92c88c', 'Y'), (0, -25, '#80a9e9', 'Z')]
        for (dx, dy, color, label) in axes:
            p.setPen(QPen(QColor(color), 2))
            p.drawLine(cx, cy, cx + dx, cy + dy)
            p.drawText(cx + dx - 4, cy + dy - 5, label)
        if self.error:
            p.fillRect(12, 40, max(100, self.width() - 90), 62, QColor(50, 28, 27, 230))
            p.setPen(QColor('#efb4a4'))
            p.drawText(self.rect().adjusted(23, 44, -84, -self.height() + 100), Qt.TextWordWrap, self.error)

    def mousePressEvent(self, event):
        self.setFocus()
        self.last = event.position()
        self.pan = event.button() in (Qt.MiddleButton, Qt.RightButton) or bool(event.modifiers() & Qt.ShiftModifier)
        self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, event):
        self.last = None
        self.unsetCursor()

    def mouseMoveEvent(self, event):
        if self.last is None or self.camera is None:
            return
        (dx, dy) = (event.position().x() - self.last.x(), event.position().y() - self.last.y())
        self.last = event.position()
        c = self.camera
        if self.pan:
            (y, p) = (c['yaw'], c['pitch'])
            k = c['distance'] * 0.0015
            right = [math.cos(y), 0, -math.sin(y)]
            up = [-math.sin(y) * math.sin(p), math.cos(p), -math.cos(y) * math.sin(p)]
            if self.up_axis == 'Z':
                right = [right[0], -right[2], right[1]]
                up = [up[0], -up[2], up[1]]
            c['target'] = [v + (-dx * r + dy * u) * k for (v, r, u) in zip(c['target'], right, up)]
        else:
            c['yaw'] -= dx * 0.006
            c['pitch'] = max(-1.55, min(1.55, c['pitch'] + dy * 0.006))
        self.changed.emit()

    def wheelEvent(self, event):
        if self.camera:
            self.camera['distance'] = max(1e-05, min(1000000000.0, self.camera['distance'] * math.exp(-event.angleDelta().y() * 0.001)))
            self.changed.emit()

    def reset(self):
        if self.home:
            self.camera = copy.deepcopy(self.home)
            self.camera['up_axis'] = self.up_axis
            self.changed.emit()

    def set_up_axis(self, axis):
        self.up_axis = axis
        if self.camera:
            self.camera['up_axis'] = axis
        self.update()
        self.changed.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F:
            self.reset()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        self.changed.emit()
        super().resizeEvent(event)

class Workspace(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('YK Gaussian Studio')
        self.resize(1480, 940)
        self.setMinimumSize(1100, 720)
        self.setDockOptions(QMainWindow.AllowNestedDocks | QMainWindow.AllowTabbedDocks | QMainWindow.AnimatedDocks)
        self.process = None
        self.url = None
        self.busy = False
        self.dirty = False
        self.preview_config = json.loads((ROOT / 'configs/preview.json').read_text(encoding='utf-8'))
        self.last_frame_started = 0.0
        self.last_presented = 0.0
        self.last_interaction = 0.0
        self.closing = False
        self.buffer = ''
        self.saved = {}
        self.previous_task = None
        self.generation = 0
        self.network = QNetworkAccessManager(self)
        self.viewport = Viewport()
        self.viewport.changed.connect(self.mark_dirty)
        self.make_center()
        self.make_toolbar()
        self.make_docks()
        self.make_menu()
        self.statusBar().showMessage(tr('就绪'))
        self.metrics = QLabel(tr('GPU 渲染  ·  精细模型'))
        self.statusBar().addPermanentWidget(self.metrics)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.frame)
        self.timer.start(16)
        self.make_form()
        self.default_state = self.saveState()

    def button(self, text, callback, primary=False):
        b = QPushButton(text)
        b.clicked.connect(callback)
        if primary:
            b.setObjectName('primary')
        return b

    def make_center(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(12, 7, 12, 7)
        row.addWidget(QLabel(tr('3D 视口')))
        row.addStretch()
        row.addWidget(QLabel(tr('向上轴')))
        self.up_axis = QComboBox()
        self.up_axis.addItems(['Y', 'Z'])
        self.up_axis.setCurrentText('Z')
        self.up_axis.currentTextChanged.connect(self.viewport.set_up_axis)
        row.addWidget(self.up_axis)
        self.quality = QComboBox()
        self.quality.addItems([tr('640 · 快速'), tr('960 · 标准'), tr('1440 · 高质量')])
        self.quality.setCurrentIndex(1)
        self.quality.currentIndexChanged.connect(self.mark_dirty)
        row.addWidget(self.quality)
        self.preview_enabled = QCheckBox(tr('启用预览'))
        self.preview_enabled.setChecked(True)
        self.preview_enabled.setToolTip(tr('关闭后停止渲染请求，训练继续；重新勾选即可恢复。'))
        self.preview_enabled.toggled.connect(self.toggle_preview)
        row.addWidget(self.preview_enabled)
        row.addWidget(self.button(tr('复位视角  F'), self.viewport.reset))
        layout.addWidget(bar)
        layout.addWidget(self.viewport, 1)
        self.setCentralWidget(w)

    def make_toolbar(self):
        bar = QToolBar(tr('主工具栏'))
        bar.setObjectName('mainToolbar')
        bar.setMovable(False)
        self.addToolBar(bar)
        brand = QLabel('  YK  /  GAUSSIAN STUDIO  ')
        brand.setStyleSheet('color:#9dbcf1;font-weight:700;font-size:14px;')
        bar.addWidget(brand)
        bar.addSeparator()
        bar.addWidget(self.button(tr('打开场景…'), self.open_scene))
        bar.addWidget(self.button(tr('分页训练'), lambda : self.select_task(tr('分页训练'))))
        bar.addWidget(self.button(tr('导出 PLY'), lambda : self.select_task(tr('导出分页 PLY'))))
        spacer = QWidget()
        from PySide6.QtWidgets import QSizePolicy
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        self.run_button = self.button(tr('▶  运行任务'), self.start, True)
        bar.addWidget(self.run_button)
        self.stop_button = self.button(tr('■  停止'), self.stop)
        self.stop_button.setEnabled(False)
        bar.addWidget(self.stop_button)

    def dock(self, title, name, area, widget):
        d = QDockWidget(title, self)
        d.setObjectName(name)
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(area, d)
        return d

    def make_docks(self):
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.scene_item = QTreeWidgetItem(self.tree, [tr('场景集合')])
        self.model_item = QTreeWidgetItem(self.scene_item, [tr('未加载场景')])
        self.scene_item.setExpanded(True)
        group = QTreeWidgetItem(self.tree, [tr('工作流')])
        for name in TASKS:
            QTreeWidgetItem(group, [name])
        group.setExpanded(True)
        self.tree.itemClicked.connect(lambda item, col: self.select_task(item.text(0)) if item.text(0) in TASKS else None)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 10)
        ll.addWidget(self.tree)
        note = QLabel(tr('  PAGED GAUSSIAN\n\n  支持分页检查点 manifest.json\n  Resident .pt 请先转换为分页格式'))
        note.setObjectName('muted')
        note.setWordWrap(True)
        ll.addWidget(note)
        self.left_dock = self.dock(tr('场景 / 工作流'), 'sceneDock', Qt.LeftDockWidgetArea, left)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(12, 12, 12, 12)
        self.task = QComboBox()
        self.task.addItems(TASKS)
        self.task.currentTextChanged.connect(self.make_form)
        rl.addWidget(self.task)
        self.description = QLabel()
        self.description.setWordWrap(True)
        self.description.setObjectName('muted')
        rl.addWidget(self.description)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.form_host = QWidget()
        self.form = QVBoxLayout(self.form_host)
        self.form.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(self.form_host)
        rl.addWidget(scroll, 1)
        self.task_status = QLabel(tr('● 就绪'))
        rl.addWidget(self.task_status)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        rl.addWidget(self.progress)
        self.right_dock = self.dock(tr('属性 / 任务参数'), 'propertiesDock', Qt.RightDockWidgetArea, right)
        self.right_dock.setMinimumWidth(310)
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        buttons = QHBoxLayout()
        buttons.addWidget(QLabel(tr('  进程输出')))
        buttons.addStretch()
        buttons.addWidget(self.button(tr('保存日志'), self.save_log))
        buttons.addWidget(self.button(tr('清空'), lambda : self.log.clear()))
        bl.addLayout(buttons)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(10000)
        bl.addWidget(self.log)
        self.log_dock = self.dock(tr('控制台'), 'consoleDock', Qt.BottomDockWidgetArea, bottom)
        self.resizeDocks([self.left_dock, self.right_dock], [225, 330], Qt.Horizontal)
        self.resizeDocks([self.log_dock], [165], Qt.Vertical)

    def make_menu(self):
        file = self.menuBar().addMenu(tr('文件'))
        file.addAction(tr('打开分页场景…'), self.open_scene).setShortcut('Ctrl+O')
        file.addAction(tr('保存日志…'), self.save_log)
        file.addSeparator()
        file.addAction(tr('退出'), self.close)
        task = self.menuBar().addMenu(tr('任务'))
        for name in TASKS:
            task.addAction(name, lambda n=name: self.select_task(n))
        view = self.menuBar().addMenu(tr('视图'))
        view.addAction(tr('复位视角'), self.viewport.reset)
        view.addAction(tr('恢复工作区布局'), lambda : self.restoreState(self.default_state))
        help = self.menuBar().addMenu(tr('帮助'))
        help.addAction(tr('操作说明'), lambda : QMessageBox.information(self, tr('操作说明'), tr('打开分页检查点目录以加载场景。\n左键旋转，中键或 Shift+拖动平移，滚轮缩放，F 复位。\n右侧选择任务并设置参数，顶部运行。\n停止训练会丢失尚未保存的进度。\n当前版本支持查看分页检查点；.pt 可通过工作流转换。')))
        help.addAction(tr('环境诊断'), lambda : self.select_task(tr('环境诊断')))

    def select_task(self, name):
        if not self.process:
            self.task.setCurrentText(name)

    def make_form(self, *_):
        if self.previous_task is not None:
            self.saved[self.previous_task] = {k: v.text() for (k, v) in self.fields.items()}
        while self.form.count():
            item = self.form.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.fields = {}
        name = self.task.currentText()
        self.previous_task = name
        self.description.setText({tr('查看场景'): tr('直接在中央视口查看分页高斯场景。相机交互不修改检查点。'), tr('分页训练'): tr('从检查点派生新训练目录，并在此视口实时预览。'), tr('场景初始训练'): tr('从 COLMAP 数据准备初始模型。此入口包含初始层级构建流程。')}.get(name, tr('设置输入与输出后，点击顶部运行任务。')))
        for (key, label, kind) in TASKS[name][1]:
            title = QLabel(label)
            title.setObjectName('section')
            self.form.addWidget(title)
            row = QWidget()
            hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(4)
            default = kind if kind not in ('dir', 'newdir', 'json', 'pt', 'savept', 'saveply', 'savejson') else ''
            if key in ('port', 'viewer-port'):
                for port in range(8765, 8865):
                    with socket.socket() as s:
                        try:
                            s.bind(('127.0.0.1', port))
                        except OSError:
                            continue
                        default = str(port)
                        break
            if key == 'config' and name in (tr('场景初始训练'), tr('迁移为 Flat')):
                default = str(ROOT / 'configs' / ('general_balanced.json' if name == tr('场景初始训练') else 'dji_flat_90m.json'))
            field = QLineEdit(self.saved.get(name, {}).get(key, default))
            field.setPlaceholderText(label)
            self.fields[key] = field
            hl.addWidget(field)
            if kind in ('dir', 'newdir', 'json', 'pt', 'savept', 'saveply', 'savejson'):
                b = self.button('…', lambda checked=False, f=field, k=kind: self.browse(f, k))
                b.setFixedWidth(32)
                hl.addWidget(b)
            self.form.addWidget(row)
        self.form.addStretch()

    def browse(self, field, kind):
        if kind == 'dir':
            value = QFileDialog.getExistingDirectory(self, tr('选择目录'), field.text())
        elif kind == 'newdir':
            (value, _) = QFileDialog.getSaveFileName(self, tr('指定新的输出目录'), field.text(), tr('目录名称 (*)'), options=QFileDialog.DontConfirmOverwrite)
        elif kind.startswith('save'):
            ext = kind[4:]
            (value, _) = QFileDialog.getSaveFileName(self, tr('保存为'), field.text(), f'{ext.upper()} (*.{ext})')
            if value and (not Path(value).suffix):
                value += '.' + ext
        else:
            (value, _) = QFileDialog.getOpenFileName(self, tr('选择文件'), field.text(), tr(f'{kind.upper()} (*.{kind});;所有文件 (*)'))
        if value:
            field.setText(value)

    def open_scene(self):
        if self.process:
            QMessageBox.information(self, tr('任务运行中'), tr('请先停止当前任务，再打开其他场景。'))
            return
        path = QFileDialog.getExistingDirectory(self, tr('打开分页检查点目录'))
        if path:
            self.task.setCurrentText(tr('查看场景'))
            self.fields['checkpoint'].setText(path)
            self.start()

    def command(self):
        name = self.task.currentText()
        args = [TASKS[name][0]]
        if name == tr('迁移为 Flat'):
            args.append('migrate')
        elif name == tr('导出 Resident PLY'):
            args.append('export')
        for (key, label, kind) in TASKS[name][1]:
            value = self.fields[key].text().strip()
            if not value:
                if tr('可选') not in label:
                    raise ValueError(tr('请填写：') + label)
                continue
            if key in ('steps', 'port', 'viewer-port', 'camera-limit', 'iterations', 'coarse_iterations'):
                if int(value) <= 0 or (key in ('port', 'viewer-port') and int(value) > 65535):
                    raise ValueError(tr('数值超出范围：') + label)
            if key == 'pool-gib' and (not (math.isfinite(float(value)) and float(value) > 0)):
                raise ValueError(tr('预览缓存必须是正数'))
            if kind in ('dir', 'json', 'pt') and (not Path(value).exists()):
                raise ValueError(tr('输入不存在：') + value)
            args.extend(['--' + key, value])
        if name == tr('分页训练'):
            args.append('--viewer')
        return args

    def start(self):
        if self.process:
            return
        try:
            args = self.command()
        except (ValueError, OverflowError) as exc:
            QMessageBox.warning(self, tr('检查参数'), str(exc))
            return
        boot = ROOT / 'portable_boot.py'
        python = str(Path(sys.executable).with_name('python.exe'))
        argv = ['-I', '-u', str(boot), *args] if boot.exists() else ['-u', str(ROOT / 'yk_gaussian.py'), *args]
        self.disconnect_view()
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        if hasattr(self, 'training_request_path') and self.training_request_path and getattr(self, 'active_step', None) in ('initial', 'train'):
            env = QProcessEnvironment.systemEnvironment()
            env.insert('YK_SAVE_REQUEST', str(self.training_request_path))
            env.insert('YK_DESKTOP_PREVIEW', '1')
            self.process.setProcessEnvironment(env)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.buffer = ''
        self.output_decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.last_log_summary = 0.0
        self.log.appendPlainText('\n> yk-gaussian ' + subprocess.list2cmdline(args))
        self.task.setEnabled(False)
        self.form_host.setEnabled(False)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.task_status.setText(tr('● 运行中 · ') + self.task.currentText())
        self.progress.setRange(0, 0)
        self.process.start(python, argv)
        if 'checkpoint' in self.fields:
            self.model_item.setText(0, Path(self.fields['checkpoint'].text()).name)
            self.model_item.setToolTip(0, self.fields['checkpoint'].text())

    def process_error(self, error):
        if self.process:
            self.log.appendPlainText(self.process.errorString())
            if error == QProcess.FailedToStart:
                self.finished(-1)

    def read_output(self):
        if not self.process:
            return
        text = self.output_decoder.decode(bytes(self.process.readAllStandardOutput()))
        self.buffer += text
        lines = re.split(r'[\r\n]', self.buffer)
        self.buffer = lines.pop()
        for line in lines:
            self.display_output_line(line)

    def display_output_line(self, line):
        if not line.strip():
            return
        match = re.search(r'Realtime viewer: (http://127\.0\.0\.1:\d+)', line)
        if match:
            self.connect_view(match.group(1))
        try:
            record = json.loads(line)
            from training_log import summary
            from ui_i18n import is_english
            readable = summary(record, is_english()) if isinstance(record, dict) else None
        except (ValueError, TypeError, KeyError, OverflowError):
            readable = None
        if readable:
            if hasattr(self, 'training_summary'):
                self.training_summary.setText(readable)
            if hasattr(self, 'detailed_log') and self.detailed_log.isChecked():
                self.log.appendPlainText(line)
            elif time.monotonic() - self.last_log_summary >= 2 or record.get('checkpoint') or record.get('growth'):
                self.log.appendPlainText(readable)
                self.last_log_summary = time.monotonic()
            if getattr(self, 'active_step', None) == 'train' and getattr(self, 'project', None):
                target = self.project['settings']['train']['config']['options'].get('iterations', 0)
                if target > 0:
                    self.progress.setRange(0, 1000)
                    self.progress.setValue(min(1000, int(record.get('image_equivalent_progress', 0) / target * 1000)))
            return
        self.log.appendPlainText(line)

    def finished(self, code, *_):
        self.read_output()
        if self.buffer:
            self.display_output_line(self.buffer)
            self.buffer = ''
        process = self.process
        self.process = None
        if process:
            process.deleteLater()
        self.disconnect_view()
        self.task.setEnabled(True)
        self.form_host.setEnabled(True)
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if code == 0 else 0)
        self.task_status.setText(tr('● 任务完成') if code == 0 else tr(f'● 已结束 · 退出码 {code}'))
        self.statusBar().showMessage(self.task_status.text())
        if self.closing:
            self.close()

    def stop(self, checked=False, confirmed=False):
        if not self.process:
            return
        if not confirmed and QMessageBox.question(self, tr('停止任务'), tr('停止训练会丢失上次检查点之后未保存的进度。确定停止？')) != QMessageBox.Yes:
            return
        pid = self.process.processId()
        if pid:
            result = subprocess.run(['taskkill.exe', '/PID', str(pid), '/T', '/F'], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode and self.process.state() != QProcess.NotRunning:
                self.closing = False
                QMessageBox.warning(self, tr('停止失败'), result.stderr.decode(errors='replace'))

    def closeEvent(self, event):
        if self.process:
            if QMessageBox.question(self, tr('退出'), tr('任务仍在运行。退出将停止任务，未保存进度会丢失。是否退出？')) == QMessageBox.Yes:
                self.closing = True
                self.stop(confirmed=True)
            event.ignore()
        else:
            event.accept()

    def disconnect_view(self):
        self.generation += 1
        self.url = None
        self.busy = False
        self.dirty = False
        self.viewport.camera = None

    def connect_view(self, url):
        self.disconnect_view()
        self.url = url
        generation = self.generation
        request = QNetworkRequest(QUrl(url + '/api/state'))
        request.setTransferTimeout(20000)
        reply = self.network.get(request)

        def ready():
            try:
                if generation != self.generation:
                    return
                if reply.error() != QNetworkReply.NoError:
                    raise RuntimeError(reply.errorString())
                state = json.loads(bytes(reply.readAll()))
                self.viewport.home = state['home']
                self.viewport.camera = copy.deepcopy(state['home'])
                self.viewport.camera['up_axis'] = self.viewport.up_axis
                self.viewport.error = ''
                self.viewport.caption = tr('透视视图  ·  Gaussian')
                self.metrics.setText(tr(f"{state['points']:,} 高斯点  |  GPU 渲染"))
                self.toggle_preview(self.preview_enabled.isChecked())
                self.mark_dirty()
                self.statusBar().showMessage(tr('场景已连接 · 在中央视口拖动以浏览'))
            except Exception as exc:
                self.view_error(str(exc))
            finally:
                reply.deleteLater()
        reply.finished.connect(ready)

    def mark_dirty(self, *_):
        self.dirty = True
        self.last_interaction = time.monotonic()

    def toggle_preview(self, enabled):
        self.viewport.setEnabled(enabled)
        self.dirty = True
        self.statusBar().showMessage(tr('预览已开启') if enabled else tr('预览已关闭，训练继续'))
        if self.url:
            request = QNetworkRequest(QUrl(self.url + '/api/preview'))
            request.setHeader(QNetworkRequest.ContentTypeHeader, 'application/json')
            request.setTransferTimeout(3000)
            reply = self.network.post(request, json.dumps({'enabled': enabled}).encode())
            reply.finished.connect(reply.deleteLater)

    def frame(self):
        if not self.preview_enabled.isChecked() or not self.url or not self.viewport.camera or self.busy or self.isMinimized():
            return
        now = time.monotonic()
        interacting = self.viewport.last is not None or now - self.last_interaction < .2
        fps = self.preview_config['target_fps'] if interacting or self.dirty else self.preview_config['idle_fps']
        if now - self.last_frame_started < 1 / max(1, fps):
            return
        self.last_frame_started = now
        self.busy = True
        self.dirty = False
        generation = self.generation
        width = [640, 960, 1440][self.quality.currentIndex()]
        if interacting:
            width = min(width, self.preview_config['interaction_width'])
        height = max(64, min(1080, round(width * self.viewport.height() / max(1, self.viewport.width()))))
        data = dict(self.viewport.camera, width=width, height=height)
        request = QNetworkRequest(QUrl(self.url + '/api/render'))
        request.setHeader(QNetworkRequest.ContentTypeHeader, 'application/json')
        request.setTransferTimeout(60000)
        reply = self.network.post(request, json.dumps(data).encode())

        def rendered():
            try:
                if generation != self.generation:
                    return
                body = bytes(reply.readAll())
                if reply.attribute(QNetworkRequest.HttpStatusCodeAttribute) == 503:
                    self.viewport.error = body.decode('utf-8', errors='replace')
                    self.viewport.update()
                    return
                if reply.error() != QNetworkReply.NoError:
                    raise RuntimeError(body.decode('utf-8', errors='replace') or reply.errorString())
                image = QImage.fromData(body)
                if image.isNull():
                    raise RuntimeError(tr('渲染返回无效图像'))
                self.viewport.image = image
                self.viewport.error = ''
                self.viewport.update()
                ms = bytes(reply.rawHeader('X-Render-Ms')).decode()
                count = bytes(reply.rawHeader('X-Visible-Points')).decode()
                step = bytes(reply.rawHeader('X-Iteration')).decode()
                presented = time.monotonic()
                fps = 1 / (presented - self.last_presented) if self.last_presented else 0
                self.last_presented = presented
                latency = (presented - now) * 1000
                self.metrics.setText(tr(f'{int(count):,} 可见点  |  {ms} ms  |  步 {step}') + f'  |  {fps:.1f} FPS  |  RTT {latency:.0f} ms')
            except Exception as exc:
                self.view_error(str(exc))
            finally:
                if generation == self.generation:
                    self.busy = False
                reply.deleteLater()
        reply.finished.connect(rendered)

    def view_error(self, message):
        self.viewport.error = message
        self.viewport.update()
        self.preview_enabled.setChecked(False)
        self.log.appendPlainText(tr('[视口] ') + message)

    def save_log(self):
        (path, _) = QFileDialog.getSaveFileName(self, tr('保存日志'), '', tr('日志 (*.log)'))
        if path:
            Path(path).write_text(self.log.toPlainText(), encoding='utf-8')

def main():
    app = QApplication(sys.argv[:1])
    app.setApplicationName('YK Gaussian Studio')
    app.setStyle('Fusion')
    app.setStyleSheet(STYLE)
    from workflow_ui import WorkflowWorkspace
    window = WorkflowWorkspace()
    from studio_control import install_control
    window.control_server = install_control(window)
    window.show()
    sys.exit(app.exec())
if __name__ == '__main__':
    main()
