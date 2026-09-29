"""Small file protocol for saving a running trainer at a completed step."""
import json
import os
from pathlib import Path
import time

_pending = None

def checkpoint_requested():
    global _pending
    name = os.environ.get('YK_SAVE_REQUEST')
    if not name:
        return False
    path = Path(name)
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        request_id = value.get('request_id', value.get('id'))
        if request_id is None:
            request_id = path.stat().st_mtime_ns
        ack = Path(str(path) + '.ack')
        if ack.is_file() and json.loads(ack.read_text(encoding='utf-8')).get('request_id') == request_id:
            return False
        _pending = (path, request_id)
        return True
    except (OSError, ValueError):
        return False

def acknowledge_checkpoint(checkpoint, iteration, stage):
    global _pending
    if _pending is None:
        return
    path, request_id = _pending
    ack = Path(str(path) + '.ack')
    temporary = Path(str(ack) + '.tmp')
    temporary.write_text(json.dumps(dict(request_id=request_id, checkpoint=str(checkpoint),
        iteration=int(iteration), stage=stage, saved_at=time.time())), encoding='utf-8')
    os.replace(temporary, ack)
    _pending = None
