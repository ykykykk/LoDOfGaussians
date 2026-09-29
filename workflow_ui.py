"""Ordered, persistent project workspace on top of the native viewport."""
from ui_i18n import tr
from ui_widgets import WheelSafeComboBox
import copy
import json
import math
from pathlib import Path
import socket
import tempfile
import uuid
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QColor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QToolBar, QTreeWidget, QTreeWidgetItem, QComboBox, QLineEdit, QCheckBox, QScrollArea, QTabWidget, QPlainTextEdit, QProgressBar, QFileDialog, QMessageBox, QSizePolicy
from desktop_ui import Workspace, ROOT
from workflow_project import STEPS, STEP_TITLES, create_project, load_project, save_project, update_settings, can_run, current_checkpoint, result_available
STATUS = {'pending': tr('待处理'), 'completed': tr('已完成'), 'stale': tr('需要更新'), 'running': tr('运行中'), 'failed': tr('失败'), 'interrupted': tr('已中断')}
NOTES = {'import': tr('建立项目并关联已去畸变照片与 COLMAP 标定。可选导入已有检查点，继续已有工作。'), 'check': tr('检查相机、照片、尺寸和缺失文件，生成数据检查报告。'), 'initial': tr('构建 scaffold 并执行初始 Resident 训练，产生后续精细模型准备所需的完整检查点。'), 'prepare': tr('将初始检查点准备为精细分页模型；后续训练继承实际检查点参数。'), 'train': tr('设置训练日程、学习率、增点与显存参数。每次运行生成独立目录，保留输入。'), 'evaluate': tr('在中央视口检查场景，或运行留出视角质量评估。'), 'export': tr('把当前训练结果导出为标准 Gaussian PLY。')}
LABELS = {'source_path': tr('COLMAP 数据目录'), 'output_root': tr('项目结果目录'), 'checkpoint': tr('已有检查点（可选）'), 'resolution': tr('图像缩放倍数'), 'llff_hold': tr('每 N 张留出一张评估'), 'seed': tr('随机种子'), 'coarse_iterations': tr('粗训练步数'), 'iterations': tr('训练日程总步数'), 'steps': tr('本次步数上限（0 = 完整日程）'), 'cap_max': tr('高斯 / 节点上限'), 'block_size': tr('每磁盘块点数'), 'radius_bands': tr('按支撑半径分组'), 'position_lr_init': tr('位置初始学习率'), 'position_lr_final': tr('位置最终学习率'), 'position_lr_delay_mult': tr('位置学习率延迟倍率'), 'position_lr_max_steps': tr('位置学习率衰减步数'), 'lr_multiplier': tr('学习率倍率'), 'feature_lr': tr('颜色 / SH 学习率'), 'opacity_lr': tr('不透明度学习率'), 'scaling_lr': tr('尺度学习率'), 'rotation_lr': tr('旋转学习率'), 'lambda_dssim': tr('SSIM 损失权重'), 'densification_interval': tr('增点间隔'), 'densify_from_iter': tr('增点开始步数'), 'densify_until_iter': tr('增点截止步数'), 'densify_grad_threshold': tr('增点梯度阈值'), 'densify_max_new_nodes': tr('每次增点上限'), 'densify_max_leaf_fraction': tr('每轮分裂叶点比例上限'), 'pool_gib': tr('GPU 缓存预算（GiB）'), 'headroom_gib': tr('渲染显存预留（GiB）'), 'image_cache_gib': tr('图像缓存（GiB）'), 'checkpoint_every': tr('检查点保存间隔'), 'tile_size': tr('图像裁块尺寸'), 'halo': tr('SSIM 边缘像素'), 'tiles_per_camera': tr('每相机裁块数'), 'decode_workers': tr('图像解码线程数'), 'growth_backend': tr('增点后端'), 'balanced_tiles': tr('均衡裁块'), 'camera_limit': tr('评估相机数量'), 'filename': tr('导出文件名'), 'coarse_image_cache_gib': tr('粗训练图像缓存（GiB）'), 'coarse_compact_images': tr('粗训练字节图像缓存'), 'coarse_fused_ssim': tr('粗训练融合 SSIM'), 'data_workers': tr('数据读取线程数'), 'pin_memory': tr('锁页内存'), 'data_prefetch_factor': tr('数据预取倍数'), 'profile_every': tr('性能记录间隔'), 'native_ops': tr('原生 CUDA 后端'), 'compact_images': tr('字节图像缓存'), 'adaptive_pool': tr('自适应显存池'), 'SH_degree': tr('球谐阶数'), 'exposure_lr_init': tr('曝光初始学习率'), 'exposure_lr_final': tr('曝光最终学习率'), 'exposure_lr_delay_steps': tr('曝光学习率延迟步数'), 'exposure_lr_delay_mult': tr('曝光延迟倍率'), 'preview': tr('训练实时预览'), 'viewer': tr('训练实时预览'), 'viewer_port': tr('预览端口'), 'train_from_latest': tr('从最新训练结果继续'), 'resume': tr('继续上一次训练'), 'resume_latest': tr('从最新训练结果继续'), 'reuse_model': tr('复用导入模型（跳过训练）'), 'capacity_rows': tr('GPU 缓存行数（覆盖 GiB 预算）'), 'densify_score_space': tr('增点评分空间')}
BASIC = {'import': None, 'check': None, 'initial': {'resolution', 'seed', 'preview', 'viewer_port', 'config.coarse_iterations', 'config.iterations', 'config.cap_max', 'config.resident.checkpoint_every'}, 'prepare': None, 'train': {'steps', 'preview', 'viewer', 'viewer_port', 'resume_latest', 'reuse_model', 'train_from_latest', 'resume', 'config.options.iterations', 'config.options.cap_max', 'config.options.densify_grad_threshold', 'config.paged.pool_gib', 'config.paged.headroom_gib', 'config.paged.checkpoint_every'}, 'evaluate': None, 'export': None}
TRAIN_OPTIONS = {'iterations', 'cap_max', 'position_lr_init', 'position_lr_final', 'position_lr_delay_mult', 'position_lr_max_steps', 'feature_lr', 'opacity_lr', 'scaling_lr', 'rotation_lr', 'lr_multiplier', 'lambda_dssim', 'densify_from_iter', 'densify_until_iter', 'densification_interval', 'densify_grad_threshold', 'densify_max_new_nodes', 'densify_max_leaf_fraction', 'densify_score_space'}
PAGED_OPTIONS = {'tile_size', 'halo', 'tiles_per_camera', 'balanced_tiles', 'decode_workers', 'pool_gib', 'headroom_gib', 'image_cache_gib', 'checkpoint_every', 'growth_backend', 'capacity_rows'}

def leaves(obj, prefix=''):
    for (key, value) in obj.items():
        path = prefix + '.' + key if prefix else key
        if isinstance(value, dict):
            yield from leaves(value, path)
        else:
            yield (path, value)

def set_value(obj, path, value):
    keys = path.split('.')
    for key in keys[:-1]:
        obj = obj[key]
    obj[keys[-1]] = value

class WorkflowWorkspace(Workspace):

    def __init__(self):
        self.project = None
        self.auto_running = False
        self.step = 'import'
        self.editors = {}
        self.fields = {}
        self.active_step = None
        self.auxiliary = None
        self.custom_args = None
        self.loading_form = False
        self.training_request_path = None
        self.save_request_id = None
        super().__init__()
        self.save_timer = QTimer(self)
        self.save_timer.setInterval(250)
        self.save_timer.timeout.connect(self.check_saved_checkpoint)
        self.setWindowTitle(tr('YK Gaussian Studio · 项目工作流'))
        self.resizeDocks([self.left_dock, self.right_dock], [250, 380], Qt.Horizontal)
        self.refresh()

    def make_toolbar(self):
        bar = QToolBar(tr('项目工具栏'))
        bar.setObjectName('projectToolbar')
        bar.setMovable(False)
        self.addToolBar(bar)
        brand = QLabel(tr('  YK  /  项目工作流  '))
        brand.setStyleSheet('color:#a9c2e8;font-weight:700;font-size:14px')
        bar.addWidget(brand)
        bar.addSeparator()
        bar.addWidget(self.button(tr('新建项目'), self.new_project))
        bar.addWidget(self.button(tr('打开项目…'), self.open_project))
        bar.addWidget(self.button(tr('保存'), self.save_project_and_training))
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        self.run_button = self.button(tr('运行当前步骤'), self.start, True)
        bar.addWidget(self.run_button)
        self.run_all_button = self.button(tr('一键运行到底'), self.run_all)
        bar.addWidget(self.run_all_button)
        self.stop_button = self.button(tr('停止'), self.stop)
        self.stop_button.setEnabled(False)
        bar.addWidget(self.stop_button)

    def make_center(self):
        super().make_center()
        banner = QWidget()
        row = QVBoxLayout(banner)
        row.setContentsMargins(16, 10, 16, 10)
        self.step_heading = QLabel(tr('01  /  创建项目与导入数据'))
        self.step_heading.setStyleSheet('font-size:17px;font-weight:600;color:#dbe5f5;')
        row.addWidget(self.step_heading)
        self.step_note = QLabel(NOTES['import'])
        self.step_note.setWordWrap(True)
        self.step_note.setObjectName('muted')
        row.addWidget(self.step_note)
        self.centralWidget().layout().insertWidget(0, banner)
        self.viewport.caption = tr('项目视图')

    def make_docks(self):
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([tr('流程步骤'), tr('状态')])
        self.tree.setColumnWidth(0, 172)
        self.step_items = {}
        for (i, step) in enumerate(STEPS):
            item = QTreeWidgetItem(self.tree, [f'{i + 1:02d}  {tr(STEP_TITLES[step])}', tr('待处理')])
            item.setData(0, Qt.UserRole, step)
            self.step_items[step] = item
        self.tree.itemClicked.connect(lambda item, _: self.select_step(item.data(0, Qt.UserRole)))
        self.model_item = QTreeWidgetItem(self.tree, [tr('当前模型'), '—'])
        self.model_item.setFlags(self.model_item.flags() & ~Qt.ItemIsSelectable)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.tree)
        self.project_info = QLabel(tr('尚未创建项目'))
        self.project_info.setWordWrap(True)
        self.project_info.setObjectName('muted')
        self.project_info.setMargin(12)
        ll.addWidget(self.project_info)
        self.left_dock = self.dock(tr('项目 / 处理顺序'), 'workflowDock', Qt.LeftDockWidgetArea, left)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(10, 10, 10, 10)
        self.task = WheelSafeComboBox()
        self.task.addItems([tr(STEP_TITLES[s]) for s in STEPS])
        self.task.hide()
        self.description = QLabel('')
        self.description.setWordWrap(True)
        self.description.setObjectName('muted')
        rl.addWidget(self.description)
        self.form_host = QWidget()
        host = QVBoxLayout(self.form_host)
        host.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        host.addWidget(self.tabs)
        self.layouts = {}
        for (title, key) in [(tr('基础参数'), 'basic'), (tr('高级参数'), 'advanced')]:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            content = QWidget()
            content.setObjectName('parameterPage')
            layout = QVBoxLayout(content)
            layout.setContentsMargins(8, 8, 8, 8)
            scroll.setWidget(content)
            self.tabs.addTab(scroll, title)
            self.layouts[key] = layout
        rl.addWidget(self.form_host, 1)
        self.outputs = QPlainTextEdit()
        self.outputs.setReadOnly(True)
        self.outputs.setMaximumHeight(110)
        self.outputs.setPlaceholderText(tr('步骤结果将显示在此处'))
        rl.addWidget(self.outputs)
        self.preview_button = self.button(tr('在视口中查看当前模型'), self.preview_model)
        rl.addWidget(self.preview_button)
        self.task_status = QLabel(tr('● 就绪'))
        self.task_status.setWordWrap(True)
        rl.addWidget(self.task_status)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 7)
        rl.addWidget(self.progress)
        nav = QHBoxLayout()
        self.prev_button = self.button(tr('← 上一步'), lambda : self.move_step(-1))
        self.next_button = self.button(tr('下一步 →'), lambda : self.move_step(1))
        nav.addWidget(self.prev_button)
        nav.addWidget(self.next_button)
        rl.addLayout(nav)
        self.right_dock = self.dock(tr('当前步骤 / 参数'), 'parametersDock', Qt.RightDockWidgetArea, right)
        self.right_dock.setMinimumWidth(350)
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        bar.addWidget(QLabel(tr('  处理记录')))
        bar.addStretch()
        self.detailed_log = QCheckBox(tr('详细日志'))
        bar.addWidget(self.detailed_log)
        bar.addWidget(self.button(tr('保存日志'), self.save_log))
        bl.addLayout(bar)
        self.training_summary = QLabel(tr('等待训练进度'))
        self.training_summary.setWordWrap(True)
        self.training_summary.setStyleSheet('padding: 6px; color: #a9c2e8;')
        bl.addWidget(self.training_summary)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(10000)
        bl.addWidget(self.log)
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText('Command: status | preview off | run --step initial | help')
        self.command_input.returnPressed.connect(self.execute_console_command)
        bl.addWidget(self.command_input)
        self.log_dock = self.dock(tr('进度 / 日志'), 'workflowConsole', Qt.BottomDockWidgetArea, bottom)
        self.resizeDocks([self.log_dock], [145], Qt.Vertical)

    def make_menu(self):
        menu = self.menuBar().addMenu(tr('文件'))
        menu.addAction(tr('新建项目…'), self.new_project).setShortcut('Ctrl+N')
        menu.addAction(tr('打开项目…'), self.open_project).setShortcut('Ctrl+O')
        menu.addAction(tr('保存项目'), self.save_project_and_training).setShortcut('Ctrl+S')
        menu.addSeparator()
        menu.addAction(tr('退出'), self.close)
        flow = self.menuBar().addMenu(tr('工作流程'))
        for step in STEPS:
            flow.addAction(tr(STEP_TITLES[step]), lambda s=step: self.select_step(s))
        view = self.menuBar().addMenu(tr('视图'))
        view.addAction(tr('预览当前模型'), self.preview_model)
        view.addAction(tr('复位视角'), self.viewport.reset)
        view.addAction(tr('恢复布局'), lambda : self.restoreState(self.default_state))
        help = self.menuBar().addMenu(tr('帮助'))
        help.addAction(tr('环境诊断'), self.doctor)
        help.addAction(tr('工作流说明'), lambda : QMessageBox.information(self, tr('工作流'), tr('按左侧 01—07 的顺序处理项目。\n每一步可调整基础和高级参数，运行结果会保存到项目。\n修改前置参数后，后续结果标记为需要更新；旧文件不会删除。\n导入已有检查点可跳过重新构建初始模型。\n训练与预览不能同时运行两个独立任务。')))

    def new_project(self):
        if self.process or not self.save_current():
            return
        (path, _) = QFileDialog.getSaveFileName(self, tr('新建 Gaussian 项目'), '', tr('YK 项目 (*.ykproject.json)'))
        if not path:
            return
        if not path.endswith('.ykproject.json'):
            path += '.ykproject.json'
        if Path(path).exists():
            QMessageBox.warning(self, tr('项目已存在'), tr('请选择一个新文件名；要继续已有项目请使用打开项目。'))
            return
        try:
            self.project = create_project(path)
            self.step = 'import'
            self.clear_scene()
            self.make_form()
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, tr('创建失败'), str(exc))

    def open_project(self):
        if self.process or not self.save_current():
            return
        (path, _) = QFileDialog.getOpenFileName(self, tr('打开 Gaussian 项目'), '', tr('YK 项目 (*.ykproject.json);;JSON (*.json)'))
        if not path:
            return
        try:
            project = load_project(path)
            self.project = project
            self.step = next((s for s in STEPS if project['steps'][s]['status'] != 'completed'), 'evaluate')
            self.clear_scene()
            self.make_form()
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, tr('打开失败'), str(exc))

    def clear_scene(self):
        self.disconnect_view()
        self.viewport.image = QImage()
        self.viewport.error = ''
        self.viewport.update()

    def make_form(self, *_):
        self.loading_form = True
        self.editors = {}
        self.fields = {}
        for layout in self.layouts.values():
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().hide()
                    item.widget().deleteLater()
        if not self.project:
            note = QLabel(tr('新建或打开项目，开始第一步。\n项目会保存各阶段参数与结果。'))
            note.setWordWrap(True)
            self.layouts['basic'].addWidget(note)
            self.layouts['basic'].addWidget(self.button(tr('新建项目…'), self.new_project, True))
            self.layouts['basic'].addStretch()
            self.loading_form = False
            return
        settings = self.project['settings'][self.step]
        self.training_mode = None
        if self.step == 'train':
            self.layouts['basic'].addWidget(QLabel(tr('训练方式')))
            self.training_mode = WheelSafeComboBox()
            self.training_mode.addItem(tr('继续训练（默认）'), 'continue')
            self.training_mode.addItem(tr('从准备好的模型重新开始本轮训练'), 'restart')
            self.training_mode.addItem(tr('跳过训练，直接使用导入模型'), 'reuse')
            mode = 'reuse' if settings.get('reuse_model') else 'continue' if settings.get('resume_latest', True) else 'restart'
            self.training_mode.setCurrentIndex(self.training_mode.findData(mode))
            available = bool(self.project['settings']['import'].get('checkpoint'))
            self.training_mode.model().item(2).setEnabled(available)
            self.training_mode.currentIndexChanged.connect(self.parameter_changed)
            self.layouts['basic'].addWidget(self.training_mode)
            self.mode_hint = QLabel()
            self.mode_hint.setWordWrap(True)
            self.layouts['basic'].addWidget(self.mode_hint)
        for (path, value) in leaves(settings):
            key = path.split('.')[-1]
            if key == 'mask_mode':
                self.layouts['basic'].addWidget(QLabel(tr('Mask 处理')))
                editor = WheelSafeComboBox()
                editor.addItem(tr('不用 Mask'), 'none')
                editor.addItem(tr('忽略：保留三维点'), 'ignore')
                editor.addItem(tr('裁剪：匹配轮廓并剔除背景'), 'crop')
                editor.setCurrentIndex(max(0, editor.findData(value)))
                editor.setToolTip(tr('不用 Mask：完整图像训练；忽略：遮罩外不参与训练；裁剪：从粗训练开始定期剔除背景点，并清理导出结果。'))
                editor.currentIndexChanged.connect(self.parameter_changed)
                self.layouts['basic'].addWidget(editor)
                self.editors[path] = (editor, str)
                continue
            if self.step == 'train' and key in ('resume_latest', 'reuse_model'):
                continue
            basic = BASIC[self.step]
            # Preview belongs to the viewport, not the training recipe.
            if key in ('preview', 'viewer', 'viewer_port') or path.startswith('config.viewer.'):
                continue
            # Preparation inherits its model contract; training controls live in step 5.
            if self.step == 'prepare' and path.startswith('options.'):
                continue
            # These legacy switches are unused by the active Resident v2 trainer.
            if self.step == 'initial' and key in ('use_GPU_caching', 'cache_size_after_reduction', 'clear_cache_interval', 'opacity_reset_interval', 'percent_dense'):
                continue
            if self.step == 'train' and path.startswith('config.options.') and (key not in TRAIN_OPTIONS):
                continue
            if self.step == 'train' and path.startswith('config.paged.') and (key not in PAGED_OPTIONS):
                continue
            if self.step == 'initial' and (path.startswith('config.general_policy.') or key in ('SH_degree', 'training_backend', 'resident_version', 'storage_device', 'densification', 'vary_distance_multiplier', 'noise_lr', 'densify_percent', 'lambda_scaling', 'lambda_opacity', 'depth_l1_weight_init', 'depth_l1_weight_final')):
                continue
            layout = self.layouts['basic' if basic is None or path in basic else 'advanced']
            title = QLabel(LABELS.get(key, key))
            title.setObjectName('section')
            title.setToolTip(path)
            layout.addWidget(title)
            choices = {'native_ops': ('auto', 'cuda', 'torch'), 'densify_score_space': ('pixel', 'ndc'), 'growth_backend': ('resident', 'flush')}.get(key)
            if choices and isinstance(value, str):
                editor = WheelSafeComboBox()
                editor.addItems(list(choices) + ([value] if value not in choices else []))
                editor.setCurrentText(value)
                editor.currentTextChanged.connect(self.parameter_changed)
                layout.addWidget(editor)
            elif isinstance(value, bool):
                editor = QCheckBox(tr('启用'))
                editor.setChecked(value)
                editor.toggled.connect(self.parameter_changed)
                layout.addWidget(editor)
            else:
                row = QWidget()
                hl = QHBoxLayout(row)
                hl.setContentsMargins(0, 0, 0, 0)
                hl.setSpacing(4)
                text = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, type(None))) else str(value)
                editor = QLineEdit(text)
                editor.setToolTip(path)
                editor.editingFinished.connect(self.parameter_changed)
                hl.addWidget(editor)
                if key in ('source_path', 'output_root', 'checkpoint'):
                    b = self.button('…', lambda checked=False, e=editor, k=key: self.browse_parameter(e, k))
                    b.setFixedWidth(32)
                    hl.addWidget(b)
                layout.addWidget(row)
            self.editors[path] = (editor, type(value))
            if self.step == 'initial' and self.project['steps']['import']['outputs'].get('checkpoint'):
                editor.setEnabled(False)
                editor.setToolTip(tr('已导入检查点，本步骤复用模型，不重新训练。'))
            if self.step == 'prepare' and key in ('block_size', 'radius_bands') and self.project['steps']['initial']['outputs'].get('checkpoint', '').endswith('manifest.json'):
                editor.setEnabled(False)
                editor.setToolTip(tr('已有分页检查点保留原磁盘块布局。'))
        for layout in self.layouts.values():
            layout.addStretch()
        self.loading_form = False
        self.task.setCurrentText(tr(STEP_TITLES[self.step]))
        self.refresh()

    def browse_parameter(self, editor, key):
        if key == 'checkpoint':
            answer = QMessageBox.question(self, tr('检查点类型'), tr('选择分页检查点目录？\n选“否”则选择 Resident .pt 文件。'), QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if answer == QMessageBox.Cancel:
                return
            if answer == QMessageBox.No:
                (path, _) = QFileDialog.getOpenFileName(self, tr('Resident 检查点'), '', tr('检查点 (*.pt)'))
            else:
                path = QFileDialog.getExistingDirectory(self, tr('分页检查点目录'))
        else:
            path = QFileDialog.getExistingDirectory(self, tr('选择目录'), editor.text())
        if path:
            editor.setText(path)
            self.parameter_changed()

    def collect(self):
        result = copy.deepcopy(self.project['settings'][self.step])
        for (path, (editor, kind)) in self.editors.items():
            if isinstance(editor, QComboBox):
                value = editor.currentData() if path == 'mask_mode' else editor.currentText()
            elif kind is bool:
                value = editor.isChecked()
            else:
                text = editor.text().strip()
                try:
                    if kind is int:
                        value = int(text)
                    elif kind is float:
                        value = float(text)
                        if not math.isfinite(value):
                            raise ValueError()
                    elif kind is str:
                        value = text
                    else:
                        value = json.loads(text)
                except (ValueError, TypeError):
                    raise ValueError(tr('参数格式无效：') + LABELS.get(path.split('.')[-1], path))
            set_value(result, path, value)
        if self.step == 'train' and self.training_mode is not None:
            mode = self.training_mode.currentData()
            result['resume_latest'] = mode == 'continue'
            result['reuse_model'] = mode == 'reuse'
        return result

    def parameter_changed(self, *_):
        if self.loading_form or not self.project or self.process:
            return
        try:
            if update_settings(self.project, self.step, self.collect()):
                save_project(self.project)
            self.refresh()
        except Exception as exc:
            from workflow_project import ProjectChangedError
            if isinstance(exc, ProjectChangedError):
                self.project = load_project(self.project['path'])
                self.make_form()
                self.refresh()
            self.task_status.setText(str(exc))

    def save_project_and_training(self, *_):
        if not self.process:
            saved = self.save_current()
            if saved:
                self.statusBar().showMessage(tr('项目已保存'))
            return saved
        if self.active_step not in ('initial', 'train') or not self.training_request_path:
            self.statusBar().showMessage(tr('当前任务没有可保存的训练状态'))
            return False
        if self.save_request_id:
            return True
        self.save_request_id = uuid.uuid4().hex
        path = Path(self.training_request_path)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'id': self.save_request_id}), encoding='utf-8')
        temporary.replace(path)
        self.statusBar().showMessage(tr('正在请求保存检查点，请等待训练完成当前步…'))
        self.log.appendPlainText(tr('正在请求保存检查点，请等待训练完成当前步…'))
        self.save_timer.start()
        return True

    def check_saved_checkpoint(self):
        if not self.save_request_id or not self.training_request_path:
            return
        ack = Path(str(self.training_request_path) + '.ack')
        try:
            saved = json.loads(ack.read_text(encoding='utf-8'))
            if saved.get('request_id') != self.save_request_id:
                return
            if not Path(saved['checkpoint']).is_file():
                return
        except (OSError, ValueError, KeyError):
            return
        message = tr('训练检查点已保存') + f": {saved['stage']} / {saved['iteration']} / {saved['checkpoint']}"
        self.log.appendPlainText(message)
        self.statusBar().showMessage(message)
        self.save_request_id = None
        self.save_timer.stop()

    def save_current(self, *_):
        if self.loading_form:
            return True
        if not self.project:
            return True
        if self.process:
            return True
        try:
            update_settings(self.project, self.step, self.collect())
            save_project(self.project)
            self.refresh()
            return True
        except Exception as exc:
            from workflow_project import ProjectChangedError
            if isinstance(exc, ProjectChangedError):
                self.project = load_project(self.project['path'])
                self.make_form()
                self.refresh()
            QMessageBox.warning(self, tr('无法保存参数'), str(exc))
            return False

    def execute_console_command(self):
        from studio_control import dispatch
        import shlex
        command = self.command_input.text().strip()
        if not command:
            return
        try:
            result = dispatch(self, shlex.split(command))
            self.log.appendPlainText('> ' + command + '\n' + json.dumps(result, ensure_ascii=False, indent=2))
            self.command_input.clear()
        except Exception as exc:
            self.log.appendPlainText(str(exc))

    def select_step(self, step):
        if step not in STEPS or self.process or step == self.step:
            return
        if not self.save_current():
            return
        self.step = step
        self.make_form()

    def move_step(self, delta):
        index = STEPS.index(self.step) + delta
        if 0 <= index < len(STEPS):
            self.select_step(STEPS[index])

    def refresh(self):
        i = STEPS.index(self.step)
        self.step_heading.setText(f'{i + 1:02d}  /  {tr(STEP_TITLES[self.step])}')
        note = NOTES[self.step]
        if self.project and self.step == 'train' and self.project['settings']['train'].get('reuse_model'):
            note = tr('本步骤将复用导入模型，不执行新的训练。')
        if self.project and self.step == 'initial' and self.project['steps']['import']['outputs'].get('checkpoint'):
            note = tr('已导入检查点。本步骤确认复用已有模型，不重新训练；完成后进入精细模型准备。')
        self.step_note.setText(note)
        self.tree.setCurrentItem(self.step_items[self.step])
        self.prev_button.setEnabled(i > 0 and (not self.process))
        self.next_button.setEnabled(bool(self.project) and i < len(STEPS) - 1 and not self.process)
        self.run_all_button.setEnabled(bool(self.project) and not self.process and not self.auto_running)
        if not self.project:
            self.run_button.setEnabled(False)
            self.preview_button.setEnabled(False)
            self.outputs.clear()
            return
        p = self.project
        done = 0
        for (step, item) in self.step_items.items():
            accessible, blocked_reason = can_run(p, step)
            item.setDisabled(False)
            item.setToolTip(0, tr(blocked_reason))
            status = 'running' if self.process and self.active_step == step else p['steps'][step]['status']
            done += status == 'completed'
            label = tr('已复用') if status == 'completed' and p['steps'][step]['outputs'].get('reused') else STATUS.get(status, status)
            item.setText(1, label)
            item.setForeground(1, QColor('#90caa3' if status == 'completed' else '#d5ae73' if status == 'stale' else '#e89898' if status in ('failed', 'interrupted') else '#a8b1c0'))
        self.project_info.setText(str(p.get('name', Path(p['path']).stem)) + '\n\n' + p['path'])
        self.setWindowTitle(str(p.get('name', tr('项目'))) + ' — YK Gaussian Studio')
        (ready, reason) = can_run(p, self.step)
        reason = tr(reason)
        self.run_button.setEnabled(ready and (not self.process))
        resume = False
        if self.step in ('initial', 'train') and not self.process:
            from workflow_runner import resume_identity, resumable_path
            signature, _ = resume_identity(p, self.step)
            resume = bool(resumable_path(p['steps'][self.step], signature, self.step))
        self.run_button.setText(tr('继续当前步骤') if resume else tr('运行当前步骤'))
        self.run_button.setToolTip(reason or tr('运行当前步骤'))
        self.description.setText(tr('参数修改后自动保存。高级参数会传给对应训练后端。'))
        if self.step == 'initial' and 'config.view_graph_k' in self.editors:
            self.editors['config.view_graph_k'][0].setEnabled(bool(p['settings']['initial']['config'].get('graph_view_select')) and not p['steps']['import']['outputs'].get('checkpoint'))
        if self.step == 'train':
            reuse = p['settings']['train'].get('reuse_model', False)
            for path, (editor, _) in self.editors.items():
                editor.setEnabled(not reuse)
            if self.training_mode is not None:
                messages = {'continue': tr('优先接着最近检查点训练；没有检查点时使用准备好的模型。'), 'restart': tr('使用第 4 步模型开始新一轮训练，保留旧结果。'), 'reuse': tr('不执行优化，直接进入评估和导出。')}
                self.mode_hint.setText(messages[self.training_mode.currentData()])
        if not self.process:
            self.progress.setRange(0, len(STEPS))
            self.progress.setValue(done)
            status = p['steps'][self.step]
            self.task_status.setText(STATUS.get(status['status'], status['status']) + (' · ' + reason if not ready else tr(' · 可以运行')))
        outputs = p['steps'][self.step].get('outputs', {})
        details = json.dumps(outputs, ensure_ascii=False, indent=2) if outputs else ''
        error = p['steps'][self.step].get('error', '')
        if error:
            details = error + '\n\n' + details
        if self.step == 'check' and outputs.get('report'):
            try:
                report = json.loads(Path(outputs['report']).read_text(encoding='utf-8'))
                details = tr(f"相机：{report['cameras']}  照片：{report['images']}\n训练视角：{report['training_views']}  评估视角：{report['test_views']}\n训练尺寸：{report['training_sizes']}\n") + details
            except (OSError, ValueError, KeyError):
                pass
        self.outputs.setPlainText(details)
        if outputs.get('reused'):
            self.step_items[self.step].setText(1, tr('已复用'))
        checkpoint = self.current_checkpoint()
        self.preview_button.setEnabled(bool(checkpoint) and (not self.process))
        if checkpoint:
            self.model_item.setText(0, tr('模型 · ') + Path(checkpoint).name)
            self.model_item.setToolTip(0, checkpoint)
        else:
            self.model_item.setText(0, tr('当前模型'))
            self.model_item.setToolTip(0, '')

    def current_checkpoint(self):
        if not self.project:
            return None
        value = current_checkpoint(self.project)
        if value:
            p = Path(value)
            p = p.parent if p.name == 'manifest.json' else p
            if (p / 'manifest.json').is_file():
                return str(p)
        return None

    def command(self):
        if self.custom_args:
            return self.custom_args
        return ['workflow', '--project', self.project['path'], '--step', self.step]

    def run_all(self):
        if self.process or not self.project or not self.save_current():
            return
        from workflow_runner import validate_settings
        try:
            for step in STEPS:
                if not result_available(self.project, step):
                    validate_settings(step, self.project['settings'][step])
        except Exception as exc:
            self.statusBar().showMessage(str(exc))
            self.log.appendPlainText(str(exc))
            return
        self.auto_running = True
        self.run_next_pending()

    def run_next_pending(self):
        if not self.auto_running or self.process:
            return
        pending = next((step for step in STEPS if not result_available(self.project, step)), None)
        if pending is None:
            self.auto_running = False
            self.refresh()
            self.statusBar().showMessage(tr('全部步骤已完成'))
            return
        ready, reason = can_run(self.project, pending)
        if not ready:
            self.auto_running = False
            self.refresh()
            self.statusBar().showMessage(tr(reason))
            return
        self.step = pending
        self.make_form()
        self.start()
        if not self.process:
            self.auto_running = False
            self.refresh()

    def stop(self, checked=False, confirmed=False):
        if self.process and not confirmed:
            if QMessageBox.question(self, tr('停止任务'), tr('停止训练会丢失上次检查点之后未保存的进度。确定停止？')) != QMessageBox.Yes:
                return
        self.auto_running = False
        super().stop(checked, confirmed=True)

    def start(self):
        if self.process or not self.project or (not self.save_current()):
            return
        (ready, reason) = can_run(self.project, self.step)
        reason = tr(reason)
        if not ready:
            QMessageBox.information(self, tr('前置步骤未完成'), reason)
            return
        self.training_request_path = Path(tempfile.gettempdir()) / ('yk-save-' + uuid.uuid4().hex + '.json')
        self.save_request_id = None
        self.active_step = self.step
        self.auxiliary = None
        self.custom_args = None
        super().start()
        self.statusBar().showMessage(tr('● 运行中 · ') + tr(STEP_TITLES[self.step]))
        self.viewport.image = QImage()
        self.viewport.empty_title = tr('正在准备训练预览') if self.step in ('initial', 'train') else tr('等待模型结果')
        self.viewport.empty_hint = tr('数据加载完成后自动连接；训练时可旋转、平移和缩放。')
        self.viewport.update()
        self.tree.setEnabled(False)
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.refresh()

    def preview_model(self):
        if self.process or not self.save_current():
            return
        checkpoint = self.current_checkpoint()
        if not checkpoint:
            return
        port = 0
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.custom_args = ['view', '--checkpoint', checkpoint, '--port', str(port), '--pool-gib', '4']
        self.active_step = None
        self.auxiliary = 'preview'
        super().start()
        self.tree.setEnabled(False)
        self.refresh()
        self.task_status.setText(tr('● 场景预览 · 只读，不执行训练'))

    def doctor(self):
        if self.process:
            return
        self.custom_args = ['doctor']
        self.active_step = None
        self.auxiliary = 'doctor'
        super().start()
        self.tree.setEnabled(False)
        self.task_status.setText(tr('● 环境诊断中'))

    def finished(self, code, *args):
        self.check_saved_checkpoint()
        if self.save_request_id:
            self.log.appendPlainText(tr('任务结束前未确认保存，请使用最近的自动检查点。'))
        self.save_request_id = None
        self.save_timer.stop()
        active = self.active_step
        closing = self.closing
        self.closing = False
        super().finished(code, *args)
        if self.project and active:
            try:
                self.project = load_project(self.project['path'])
                record = self.project['steps'][active]
                if code and record['status'] == 'running':
                    record['status'] = 'failed'
                    record['error'] = tr('进程已停止，未完成该步骤。')
                    save_project(self.project)
            except Exception as exc:
                self.log.appendPlainText(tr('[项目状态] ') + str(exc))
        self.custom_args = None
        self.active_step = None
        self.auxiliary = None
        self.tree.setEnabled(True)
        self.make_form()
        self.refresh()
        if self.project and active and (code == 0) and (self.project['steps'][active]['status'] == 'completed'):
            self.statusBar().showMessage(tr('当前步骤已完成。检查结果后，点击“下一步”继续。'))
        if closing:
            self.auto_running = False
            self.close()
        elif self.auto_running:
            if code == 0 and active and self.project['steps'][active]['status'] == 'completed':
                QTimer.singleShot(0, self.run_next_pending)
            else:
                self.auto_running = False
                self.statusBar().showMessage(tr('自动运行已停止，请检查当前步骤'))
                self.refresh()

    def closeEvent(self, event):
        if not self.process and (not self.save_current()):
            event.ignore()
            return
        super().closeEvent(event)
