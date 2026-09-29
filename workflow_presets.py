"""Load editable workflow parameter defaults independently of the desktop UI.

The JSON file contains all seven stages. Each call parses a fresh settings
object, so project edits cannot mutate the defaults of another project.
"""
import json
from pathlib import Path

DEFAULTS_PATH = Path(__file__).resolve().parent / 'configs' / 'workflow_defaults.json'


def load_defaults():
    """Read the short quality preset used when creating a new project."""
    settings = json.loads(DEFAULTS_PATH.read_text(encoding='utf-8-sig'))
    stages = ('import', 'check', 'initial', 'prepare', 'train', 'evaluate', 'export')
    if not isinstance(settings, dict) or any(
            not isinstance(settings.get(stage), dict) for stage in stages):
        raise ValueError(f'Workflow defaults must contain seven stage objects: {DEFAULTS_PATH}')
    return settings
