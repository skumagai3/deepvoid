"""Tests for the UNETR model and its config checks."""

from pathlib import Path

import numpy as np
import pytest

from deepvoid.config import load_config

BASE = {'name': 't', 'data': {'inputs': ['a'], 'labels': 'm'}}
SMALL = {'patch': 4, 'dim': 12, 'blocks': 2, 'heads': 2, 'mlp_ratio': 2}


def unetr_config(**sections):
    cfg = {**BASE, 'model': {'arch': 'unetr'}, **sections}
    cfg['model'] = {'arch': 'unetr', **sections.get('model', {})}
    return cfg


def test_default_arch_is_unet():
    assert load_config(BASE)['model']['arch'] == 'unet'


def test_unetr_config_checks():
    ok = unetr_config(evaluate={'crop': 96, 'margin': 24})
    assert load_config(ok)['model']['vit']['patch'] == 8
    bad = [
        unetr_config(),                                               # evaluate.crop 128 > 96
        unetr_config(training={'crop': 100}, evaluate={'crop': 96}),  # 100 not a multiple of 8
        unetr_config(model={'deep_supervision': True}, evaluate={'crop': 96}),
        unetr_config(model={'strip_pool': {'axes': ['z'], 'length': 5}}, evaluate={'crop': 96}),
        {**BASE, 'model': {'arch': 'vit'}},
    ]
    for cfg in bad:
        with pytest.raises(ValueError):
            load_config(cfg)


def test_packaged_unetr_config_is_valid():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / 'configs/research/vit/C_mixed_smooth2ch_unetr.yaml')
    assert cfg['model']['arch'] == 'unetr'


def test_tap_blocks_match_the_paper():
    pytest.importorskip('tensorflow')
    from deepvoid.models.unetr import tap_blocks
    assert tap_blocks(16, 12) == {1: 3, 2: 6, 3: 9, 4: 12}
    assert tap_blocks(8, 9) == {1: 3, 2: 6, 3: 9}


def test_position_embedding_depends_on_offset_and_not_grid_size():
    pytest.importorskip('tensorflow')
    from deepvoid.models.unetr import SinePosition3D
    pos = SinePosition3D()
    small = np.asarray(pos(np.zeros((1, 4, 4, 4, 12), np.float32)))[0]
    large = np.asarray(pos(np.zeros((1, 6, 5, 7, 12), np.float32)))[0]
    # The same token index gets the same embedding in any grid.
    assert np.allclose(small, large[:4, :4, :4], atol=1e-6)
    # Dot products depend only on the offset: (0,0,0)-(1,2,0) equals (2,1,3)-(3,3,3).
    a = small[0, 0, 0] @ small[1, 2, 0]
    b = large[2, 1, 3] @ large[3, 3, 3]
    assert np.isclose(a, b, atol=1e-5)
    with pytest.raises(ValueError):
        pos2 = SinePosition3D()
        pos2(np.zeros((1, 2, 2, 2, 8), np.float32))


def test_unetr_shapes_for_any_multiple_of_the_patch():
    pytest.importorskip('tensorflow')
    from deepvoid.models import unetr3d
    model = unetr3d(**SMALL, base_filters=4, in_channels=2)
    for n in (8, 12, 16):
        y = np.asarray(model(np.random.rand(2, n, n, n, 2).astype(np.float32)))
        assert y.shape == (2, n, n, n, 4)
        assert np.allclose(y.sum(-1), 1, atol=1e-5)


def test_build_model_dispatches_on_arch():
    pytest.importorskip('tensorflow')
    from deepvoid.models import build_model
    cfg = load_config(unetr_config(model={'vit': SMALL, 'base_filters': 4},
                                   training={'crop': 16, 'val_crop': 8},
                                   evaluate={'crop': 16, 'margin': 4}))
    assert build_model(cfg['model']).get_layer('patch_embed') is not None
    assert build_model(load_config(BASE)['model']).name == 'unet3d'


def test_unetr_save_and_load_give_the_same_scores(tmp_path):
    pytest.importorskip('tensorflow')
    from deepvoid.models import load_model, unetr3d
    model = unetr3d(**SMALL, base_filters=4)
    x = np.random.rand(1, 8, 8, 8, 1).astype(np.float32)
    path = tmp_path / 'm.keras'
    model.save(path)
    assert np.allclose(model.predict_on_batch(x), load_model(path).predict_on_batch(x),
                       atol=1e-6)


def test_unetr_trains_one_step_with_xla():
    pytest.importorskip('tensorflow')
    import keras
    from deepvoid.models import unetr3d
    model = unetr3d(**SMALL, base_filters=4)
    model.compile(optimizer='adamw', loss='sparse_categorical_crossentropy', jit_compile=True)
    x = np.random.rand(2, 8, 8, 8, 1).astype(np.float32)
    y = np.random.randint(0, 4, size=(2, 8, 8, 8))
    loss = model.train_on_batch(x, y)
    assert np.isfinite(loss)
    keras.backend.clear_session()
