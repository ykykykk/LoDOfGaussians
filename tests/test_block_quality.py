import math
import pytest
from tools.evaluate_block_quality import metrics, tile_rectangles


def test_tile_cores_cover_every_pixel_once():
    rects = list(tile_rectangles(11, 7, 4))
    pixels = [(x, y) for left, top, w, h in rects
              for x in range(left, left+w) for y in range(top, top+h)]
    assert len(pixels) == len(set(pixels)) == 77
    assert (10, 6) in pixels


def test_pixel_weighted_metrics_not_average_tile_psnr():
    result = metrics(3*.01*10 + 3*.09*2, 3*.8*10 + 3*.5*2, 12)
    assert result['mse'] == pytest.approx(.28/12)
    assert result['psnr'] == pytest.approx(-10*math.log10(.28/12))
    assert result['ssim'] == pytest.approx(.75)
    assert metrics(0, 3, 1)['perfect_match']


def test_invalid_dimensions():
    with pytest.raises(ValueError):
        list(tile_rectangles(10, 10, 0))
    with pytest.raises(ValueError):
        metrics(0, 0, 0)
