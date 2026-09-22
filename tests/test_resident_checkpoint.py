from types import SimpleNamespace
import numpy as np
import torch
import pytest
from utils.resident_checkpoint import save_checkpoint, load_checkpoint
from utils.view_pipeline import ViewSchedule


def test_checkpoint_restores_moments_scores_and_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    def model():
        return SimpleNamespace(size=3, properties=torch.randn(8, 69),
            nodes=torch.zeros(8, 6, dtype=torch.int32), _densification_criterium=torch.arange(8).float(),
            active_sh_degree=1, skybox_points=1, spatial_lr_scale=np.float32(2.))
    g = model()
    tracker = SimpleNamespace(seen=torch.ones(8, dtype=torch.bool), views=5)
    path = tmp_path/'latest.pt'
    save_checkpoint(path, g, 4, {'data': 'a'}, 7, tracker, .2, 0)
    restored = model()
    state = load_checkpoint(path, restored, {'data': 'a'}, 10)
    assert torch.equal(g.properties[:3], restored.properties[:3])
    assert torch.equal(g._densification_criterium[:3], restored._densification_criterium[:3])
    assert state['iteration'] == 4 and state['views'] == 5
    assert list(ViewSchedule(4, 11, 7, start=5)) == list(ViewSchedule(4, 11, 7))[5:]
    with pytest.raises(ValueError, match='differ'):
        load_checkpoint(path, model(), {'data': 'b'}, 10)
    def fail(*a, **k):
        raise OSError('interrupted write')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError):
        save_checkpoint(path, g, 6, {'data': 'a'}, 7, tracker, .3, 0)
    assert load_checkpoint(path, model(), {'data': 'a'}, 10)['iteration'] == 4
