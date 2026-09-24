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


def test_growth_resume_preserves_adam_but_starts_fresh_split_window(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    def model(cap):
        return SimpleNamespace(size=3, properties=torch.randn(cap, 69),
            nodes=torch.zeros(cap, 6, dtype=torch.int32),
            _densification_criterium=torch.ones(cap), active_sh_degree=1,
            skybox_points=1, spatial_lr_scale=np.float32(2.))
    options = dict(iterations=30, cap_max=8, densify_max_new_nodes=2,
                   densify_until_iter=24, densification_interval=3,
                   position_lr_max_steps=30, densify_grad_threshold=.01)
    old = dict(source='data', resolution=1, hierarchy='coarse', pipeline={}, options=options)
    new = dict(old, options=dict(options, iterations=60, cap_max=12,
                                 densify_max_new_nodes=4, densify_until_iter=48,
                                 densification_interval=5))
    original = model(8)
    original.properties[:3, 23:] = 7  # Adam columns must survive migration.
    path = tmp_path/'latest.pt'
    save_checkpoint(path, original, 29, old, 0,
                    SimpleNamespace(seen=torch.ones(8, dtype=torch.bool), views=20), .3, 2)
    with pytest.raises(ValueError, match='differ'):
        load_checkpoint(path, model(12), new, 60)
    target = model(12)
    state = load_checkpoint(path, target, new, 60, allow_growth=True)
    torch.testing.assert_close(target.properties[:3], original.properties[:3])
    assert not target._densification_criterium[:3].any()
    assert state['seen'] is None and state['views'] == state['empty_windows'] == 0
    with pytest.raises(ValueError, match='differ'):
        load_checkpoint(path, model(12), dict(new, source='other'), 60, allow_growth=True)
    with pytest.raises(ValueError, match='differ'):
        load_checkpoint(path, model(12), dict(new, options=dict(new['options'],
                        densify_grad_threshold=.001)), 60, allow_growth=True)
