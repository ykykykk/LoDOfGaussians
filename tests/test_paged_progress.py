from types import SimpleNamespace
import pytest
from utils.paged_progress import restore_progress, crossed_growth, maximum_updates
from utils.camera_tiles import tile_rectangle


def test_restore_legacy_clock_uses_actual_core_areas():
    info=SimpleNamespace(image_name='a',width=19,height=13)
    metadata={'iteration':110,'camera_visits':{'a':10},'contract':{'resolution':1}}
    expected=100+sum(tile_rectangle(19,13,'a',i,7,8,True)[2]*tile_rectangle(19,13,'a',i,7,8,True)[3]/247 for i in range(10))
    assert restore_progress(metadata,[info],8,True,7)==pytest.approx(expected)
    metadata['image_equivalent_progress']=expected
    assert restore_progress(metadata,[],8,True,7)==expected


def test_growth_fires_once_on_coverage_boundary():
    assert not crossed_growth(9.8,9.9,0,30,10)
    assert crossed_growth(9.99,10.05,0,30,10)
    assert not crossed_growth(10.05,10.1,0,30,10)
    assert not crossed_growth(29.99,30.01,0,30,10)
    assert not crossed_growth(9.99,10.01,10,30,10)


def test_update_bound_covers_small_border_tiles():
    info=SimpleNamespace(width=19,height=13)
    assert maximum_updates(1,[info],1,8,True)>=6
    assert maximum_updates(1,[info],1,8,False)>=6
