"""System-language UI translations, without importing the GUI or GPU runtime."""
import ctypes
import locale
import os
from functools import lru_cache

@lru_cache(maxsize=1)
def language():
    """Chinese on a Chinese Windows display language; English otherwise."""
    override = os.environ.get('YK_UI_LANGUAGE', '').lower()
    if override in ('zh', 'en'):
        return override
    if os.name == 'nt':
        try:
            return 'zh' if ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0x3ff == 0x04 else 'en'
        except (AttributeError, OSError):
            pass
    try:
        name = locale.getlocale()[0] or ''
    except (ValueError, TypeError):
        name = ''
    return 'zh' if name.lower().startswith(('zh', 'chinese')) else 'en'

def is_english():
    return language() == 'en'

# Source strings are stable keys. Keep punctuation and whitespace in UI fragments.
_PAIRS = '''一键运行到底|Run all remaining steps
全部步骤已完成|All steps completed
自动运行已停止，请检查当前步骤|Automatic run stopped; check the current step
详细日志|Detailed log
等待训练进度|Waiting for training progress
训练方式|Training mode
继续训练（默认）|Continue training (default)
从准备好的模型重新开始本轮训练|Start a new run from the prepared model
跳过训练，直接使用导入模型|Skip training and use the imported model
优先接着最近检查点训练；没有检查点时使用准备好的模型。|Resume the latest checkpoint, or use the prepared model if none exists.
使用第 4 步模型开始新一轮训练，保留旧结果。|Start from the step 4 model in a new run, preserving old results.
不执行优化，直接进入评估和导出。|Skip optimization and proceed to evaluation and export.
继续当前步骤|Resume current step
项目已保存|Project saved
当前任务没有可保存的训练状态|The current task has no training state to save
正在请求保存检查点，请等待训练完成当前步…|Checkpoint requested; waiting for the current training step...
训练检查点已保存|Training checkpoint saved
任务结束前未确认保存，请使用最近的自动检查点。|Save was not confirmed before task exit; use the latest automatic checkpoint.
启用预览|Enable preview
关闭后停止渲染请求，训练继续；重新勾选即可恢复。|Disable rendering requests while training continues; enable again to resume.
预览已开启|Preview enabled
预览已关闭，训练继续|Preview disabled; training continues
向上轴|Up axis
本次步数上限（0 = 完整日程）|Update limit (0 = full schedule)
正在准备训练预览|Preparing training preview
等待模型结果|Waiting for model results
数据加载完成后自动连接；训练时可旋转、平移和缩放。|Connects after data loading; orbit, pan and zoom during training.
透视视图|Perspective
就绪|Ready
GPU 渲染  ·  无 LoD|GPU rendering · Fine model
GPU 渲染  ·  精细模型|GPU rendering · Fine model
实时更新|Live update
主工具栏|Main toolbar
▶  运行任务|▶  Run task
■  停止|■  Stop
场景 / 工作流|Scene / Workflow
● 就绪|● Ready
属性 / 任务参数|Properties / Task settings
控制台|Console
文件|File
保存日志…|Save log…
退出|Exit
任务|Tasks
视图|View
复位视角|Reset view
恢复工作区布局|Restore workspace layout
帮助|Help
操作说明|Quick guide
环境诊断|System diagnostics
打开分页检查点目录|Open paged checkpoint folder
迁移为 Flat|Convert to fine model
分页训练|Paged training
保存日志|Save log
日志 (*.log)|Logs (*.log)
打开一个 Gaussian 场景|Open a Gaussian scene
选择分页检查点  ·  打开场景  ·  在此视口中交互|Select a paged checkpoint · Open scene · Explore in this viewport
3D 视口|3D viewport
640 · 快速|640 · Fast
960 · 标准|960 · Standard
1440 · 高质量|1440 · High quality
复位视角  F|Reset view  F
打开场景…|Open scene…
导出 PLY|Export PLY
场景集合|Scene collection
未加载场景|No scene loaded
工作流|Workflow
  进程输出|  Process output
清空|Clear
设置输入与输出后，点击顶部运行任务。|Set the input and output, then click Run task above.
选择目录|Select folder
任务运行中|Task running
请先停止当前任务，再打开其他场景。|Stop the current task before opening another scene.
查看场景|View scene
导出 Resident PLY|Export resident PLY
● 运行中 · |● Running · 
● 任务完成|● Task completed
透视视图  ·  Gaussian|Perspective · Gaussian
[视口] |[Viewport] 
打开分页场景…|Open paged scene…
指定新的输出目录|Choose a new output folder
目录名称 (*)|Folder name (*)
可选|Optional
预览缓存必须是正数|Preview cache must be positive
检查参数|Check parameters
● 已结束 · 退出码 |● Finished · Exit code 
停止任务|Stop task
停止训练会丢失上次检查点之后未保存的进度。确定停止？|Stopping discards unsaved progress since the last checkpoint. Stop now?
停止失败|Could not stop
任务仍在运行。退出将停止任务，未保存进度会丢失。是否退出？|A task is still running. Exiting will stop it and discard unsaved progress. Exit?
场景已连接 · 在中央视口拖动以浏览|Scene connected · Drag in the central viewport to explore
导出分页 PLY|Export paged PLY
场景初始训练|Initial scene training
直接在中央视口查看分页高斯场景。相机交互不修改检查点。|View a paged Gaussian scene in the central viewport. Camera movement does not change the checkpoint.
从检查点派生新训练目录，并在此视口实时预览。|Train from a checkpoint in a new output folder with a live viewport preview.
从 COLMAP 数据准备初始模型。此入口包含初始层级构建流程。|Prepare an initial model from COLMAP data, including the initial hierarchy construction.
保存为|Save as
选择文件|Select file
输入不存在：|Input does not exist: 
 高斯点  |  GPU 渲染| Gaussians  |  GPU rendering
渲染返回无效图像|Renderer returned an invalid image
 可见点  |  | visible points  |  
 ms  |  步 | ms  |  Step 
);;所有文件 (*)|);;All files (*)
请填写：|Please enter: 
数值超出范围：|Value out of range: 
待处理|Pending
已完成|Completed
需要更新|Needs update
运行中|Running
失败|Failed
已中断|Interrupted
建立项目并关联已去畸变照片与 COLMAP 标定。可选导入已有检查点，继续已有工作。|Create a project with undistorted photos and COLMAP calibration. Optionally import an existing checkpoint to continue previous work.
检查相机、照片、尺寸和缺失文件，生成数据检查报告。|Check cameras, photos, image dimensions and missing files, then create a dataset report.
构建 scaffold 并执行初始 Resident 训练，产生后续无 LoD 转换所需的完整检查点。|Build the scaffold and run initial resident training to create a full checkpoint for fine-model preparation.
构建 scaffold 并执行初始 Resident 训练，产生后续精细模型转换所需的完整检查点。|Build the scaffold and run initial resident training to create a full checkpoint for fine-model preparation.
将初始检查点准备为无 LoD 分页模型；后续训练继承实际检查点参数。|Prepare a paged fine model from the initial checkpoint. Subsequent training inherits its actual parameters.
将初始检查点准备为精细模型；后续训练继承实际检查点参数。|Prepare a fine model from the initial checkpoint. Subsequent training inherits its actual parameters.
将初始检查点准备为精细分页模型；后续训练继承实际检查点参数。|Prepare a paged fine model from the initial checkpoint. Subsequent training inherits its actual parameters.
设置训练日程、学习率、增点与显存参数。每次运行生成独立目录，保留输入。|Set the training schedule, learning rates, densification and GPU memory budgets. Each run creates a separate folder and preserves its input.
在中央视口检查场景，或运行留出视角质量评估。|Inspect the scene in the viewport or evaluate quality on held-out views.
把当前训练结果导出为标准 Gaussian PLY。|Export the current trained model as a standard Gaussian PLY.
COLMAP 数据目录|COLMAP dataset folder
项目结果目录|Project output folder
已有检查点（可选）|Existing checkpoint (optional)
图像缩放倍数|Image downscale factor
每 N 张留出一张评估|Hold out every Nth image
随机种子|Random seed
粗训练步数|Coarse training steps
训练日程总步数|Total training schedule steps
本次优化步数|Optimizer steps for this run
高斯 / 节点上限|Gaussian / node limit
每磁盘块点数|Points per disk block
按支撑半径分组|Group by support radius
位置初始学习率|Initial position learning rate
位置最终学习率|Final position learning rate
位置学习率延迟倍率|Position learning-rate delay multiplier
位置学习率衰减步数|Position learning-rate decay steps
学习率倍率|Learning-rate multiplier
颜色 / SH 学习率|Color / SH learning rate
不透明度学习率|Opacity learning rate
尺度学习率|Scale learning rate
旋转学习率|Rotation learning rate
SSIM 损失权重|SSIM loss weight
增点间隔|Densification interval
增点开始步数|Densification start step
增点截止步数|Densification end step
增点梯度阈值|Densification gradient threshold
每次增点上限|New points per densification
每轮分裂叶点比例上限|Maximum leaf split fraction
GPU 缓存预算（GiB）|GPU cache budget (GiB)
渲染显存预留（GiB）|Rendering memory reserve (GiB)
图像缓存（GiB）|Image cache (GiB)
检查点保存间隔|Checkpoint interval
图像裁块尺寸|Image tile size
SSIM 边缘像素|SSIM halo pixels
每相机裁块数|Tiles per camera
图像解码线程数|Image decoder threads
增点后端|Densification backend
均衡裁块|Balanced tiles
评估相机数量|Evaluation cameras
导出文件名|Export filename
粗训练图像缓存（GiB）|Coarse image cache (GiB)
粗训练字节图像缓存|Compact coarse image cache
粗训练融合 SSIM|Fused SSIM for coarse training
数据读取线程数|Data loading workers
锁页内存|Pinned memory
数据预取倍数|Data prefetch factor
性能记录间隔|Profiling interval
原生 CUDA 后端|Native CUDA backend
字节图像缓存|Compact image cache
自适应显存池|Adaptive GPU memory pool
球谐阶数|Spherical harmonics degree
曝光初始学习率|Initial exposure learning rate
曝光最终学习率|Final exposure learning rate
曝光学习率延迟步数|Exposure learning-rate delay steps
曝光延迟倍率|Exposure delay multiplier
训练实时预览|Live training preview
预览端口|Preview port
从最新训练结果继续|Resume latest training result
继续上一次训练|Continue previous training
复用导入模型（跳过训练）|Reuse imported model (skip training)
GPU 缓存行数（覆盖 GiB 预算）|GPU cache rows (overrides GiB budget)
增点评分空间|Densification score space
项目视图|Project view
YK Gaussian Studio · 项目工作流|YK Gaussian Studio · Project workflow
项目工具栏|Project toolbar
  YK  /  项目工作流  |  YK  /  Project workflow  
运行当前步骤|Run current step
停止|Stop
01  /  创建项目与导入数据|01  /  Create project and import data
尚未创建项目|No project created
项目 / 处理顺序|Project / Processing order
步骤结果将显示在此处|Step results will appear here
在视口中查看当前模型|View current model in viewport
← 上一步|← Previous
下一步 →|Next →
当前步骤 / 参数|Current step / Parameters
进度 / 日志|Progress / Log
工作流程|Workflow
预览当前模型|Preview current model
恢复布局|Restore layout
工作流说明|Workflow guide
新建 Gaussian 项目|New Gaussian project
YK 项目 (*.ykproject.json)|YK projects (*.ykproject.json)
打开 Gaussian 项目|Open Gaussian project
YK 项目 (*.ykproject.json);;JSON (*.json)|YK projects (*.ykproject.json);;JSON (*.json)
已导入检查点。本步骤确认复用已有模型，不重新训练；完成后进入无 LoD 准备。|A checkpoint is imported. This step confirms reuse without retraining; then continue to fine-model preparation.
已导入检查点。本步骤确认复用已有模型，不重新训练；完成后进入精细模型准备。|A checkpoint is imported. This step confirms reuse without retraining; then continue to fine-model preparation.
参数修改后自动保存。高级参数会传给对应训练后端。|Parameter changes are saved automatically. Advanced parameters are passed to the training backend.
● 场景预览 · 只读，不执行训练|● Scene preview · Read-only, no training
● 环境诊断中|● Running diagnostics
新建项目|New project
打开项目…|Open project…
保存|Save
流程步骤|Workflow step
状态|Status
当前模型|Current model
基础参数|Basic parameters
高级参数|Advanced parameters
  处理记录|  Processing log
项目已存在|Project already exists
请选择一个新文件名；要继续已有项目请使用打开项目。|Choose a new filename, or use Open project to continue an existing project.
检查点类型|Checkpoint type
已复用|Reused
前置步骤未完成|Previous step incomplete
当前步骤已完成。检查结果后，点击“下一步”继续。|This step is complete. Inspect the results, then click Next to continue.
新建项目…|New project…
保存项目|Save project
创建失败|Creation failed
打开失败|Open failed
启用|Enabled
已导入检查点，本步骤复用模型，不重新训练。|A checkpoint is imported. This step reuses the model without retraining.
已有分页检查点保留原磁盘块布局。|Existing paged checkpoints retain their disk block layout.
Resident 检查点|Resident checkpoint
检查点 (*.pt)|Checkpoints (*.pt)
分页检查点目录|Paged checkpoint folder
无法保存参数|Could not save parameters
模型 · |Model · 
进程已停止，未完成该步骤。|The process stopped before this step completed.
项目|Project
 · 可以运行| · Ready to run
相机：|Cameras: 
  照片：|  Photos: 
训练视角：|Training views: 
  评估视角：|  Evaluation views: 
训练尺寸：|Training dimensions: 
[项目状态] |[Project status] 
参数格式无效：|Invalid parameter format: 
导入数据|Import data
检查数据|Check data
构建初始模型|Build initial model
准备无 LoD 模型|Prepare fine model
准备精细模型|Prepare fine model
精细模型|Fine model
训练与细化|Train and refine
检查与评估|Inspect and evaluate
导出成果|Export results
已有步骤正在运行|A workflow step is already running
请先完成：|Complete this step first: 
转换为分页检查点|Convert to paged checkpoint
质量评估|Quality evaluation
预览缓存 GiB|Preview cache (GiB)
输入分页检查点|Input paged checkpoint
新输出目录|New output folder
数据目录（可选）|Dataset folder (optional)
配置（可选）|Configuration (optional)
训练配置|Training configuration
训练步数（可选）|Training steps (optional)
粗训练步数（可选）|Coarse training steps (optional)
Flat 检查点 .pt|Fine-model checkpoint .pt
Resident 检查点 .pt|Resident checkpoint .pt
新检查点 .pt|New checkpoint .pt
Flat 配置|Fine-model configuration
输出 PLY（可选）|Output PLY (optional)
输出 PLY|Output PLY
评估结果 JSON|Evaluation results JSON'''

TRANSLATIONS = dict(line.split('|', 1) for line in _PAIRS.splitlines() if '|' in line)
# Text containing a literal separator or newline is defined separately.
TRANSLATIONS.update({
    '本步骤将复用导入模型，不执行新的训练。': 'This step will reuse the imported model without running new training.',
    '构建 scaffold 并执行初始 Resident 训练，产生后续精细模型准备所需的完整检查点。': 'Build the scaffold and run initial resident training to create a full checkpoint for fine-model preparation.',
    '左键旋转   |   中键 / Shift+拖动平移   |   滚轮缩放   |   F 复位': 'Left drag: orbit   |   Middle / Shift+drag: pan   |   Wheel: zoom   |   F: reset',
    ' 高斯点  |  GPU 渲染': ' Gaussians  |  GPU rendering',
    ' 可见点  |  ': ' visible points  |  ',
    ' ms  |  步 ': ' ms  |  Step ',
    '\n训练视角：': '\nTraining views: ',
    '\n训练尺寸：': '\nTraining dimensions: ',
    '  PAGED GAUSSIAN\n\n  支持分页检查点 manifest.json\n  Resident .pt 请先转换为分页格式': '  PAGED GAUSSIAN\n\n  Supports paged manifest.json checkpoints\n  Convert Resident .pt files to paged format first',
    '新建或打开项目，开始第一步。\n项目会保存各阶段参数与结果。': 'Create or open a project to begin.\nThe project saves parameters and results for every stage.',
    '选择分页检查点目录？\n选“否”则选择 Resident .pt 文件。': 'Select a paged checkpoint folder?\nChoose No to select a Resident .pt file.',
    '按左侧 01—07 的顺序处理项目。\n每一步可调整基础和高级参数，运行结果会保存到项目。\n修改前置参数后，后续结果标记为需要更新；旧文件不会删除。\n导入已有检查点可跳过重新构建初始模型。\n训练与预览不能同时运行两个独立任务。': 'Follow steps 01–07 on the left.\nAdjust basic and advanced parameters at each step; results are saved in the project.\nChanging earlier parameters marks later results as needing an update; existing files are preserved.\nImport a checkpoint to skip rebuilding the initial model.\nTraining and preview cannot run as two separate tasks at the same time.',
    '打开分页检查点目录以加载场景。\n左键旋转，中键或 Shift+拖动平移，滚轮缩放，F 复位。\n右侧选择任务并设置参数，顶部运行。\n停止训练会丢失尚未保存的进度。\n当前版本支持查看分页检查点；.pt 可通过工作流转换。': 'Open a paged checkpoint folder to load a scene.\nLeft drag to orbit, middle or Shift+drag to pan, wheel to zoom, F to reset.\nChoose a task and parameters on the right, then run it from the toolbar.\nStopping training discards unsaved progress.\nThe viewer supports paged checkpoints; convert .pt files through the workflow.',
})

# Include terminology-renamed versions used by current GUI sources.
for _source, _english in list(TRANSLATIONS.items()):
    TRANSLATIONS.setdefault(_source.replace('准备无 LoD 模型', '准备精细模型').replace('无 LoD', '精细模型'), _english)

def tr(text):
    """Translate a source label while leaving backend keys and paths untouched."""
    if not isinstance(text, str):
        return text
    # Keep terminology current even if an older UI module still uses the old key.
    chinese = text.replace('准备无 LoD 模型', '准备精细模型').replace('无 LoD', '精细模型')
    if language() == 'zh':
        return chinese
    exact = TRANSLATIONS.get(text, TRANSLATIONS.get(chinese))
    if exact is not None:
        return exact
    result = text
    # Only known UI fragments with delimiters, never arbitrary word replacement.
    fragments = (' 高斯点  |  GPU 渲染', ' 可见点  |  ', ' ms  |  步 ', '\n训练视角：', '\n训练尺寸：', '  评估视角：', '  照片：')
    for fragment in fragments:
        result = result.replace(fragment, TRANSLATIONS[fragment])
    for prefix in ('● 运行中 · ', '● 已结束 · 退出码 ', '请先完成：', '相机：', '[项目状态] ', '[视口] ', '模型 · ', '请填写：', '输入不存在：', '数值超出范围：', '参数格式无效：'):
        if result.startswith(prefix):
            tail = result[len(prefix):]
            result = TRANSLATIONS[prefix] + TRANSLATIONS.get(tail, tail)
            break
    return result
