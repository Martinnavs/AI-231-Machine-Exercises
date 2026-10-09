import sys

import pytest

import me2_voicegen.generation.cosyvoice_env as cosyvoice_env


def test_path_resolution_is_anchored_relative_to_vendor_root(fake_project):
    env, project_root, cosyvoice_dir, matcha_dir = fake_project

    assert env.VENDOR_ROOT == project_root / "vendor"
    assert env.COSYVOICE_DIR == env.VENDOR_ROOT / "CosyVoice"
    assert env.MATCHA_TTS_DIR == env.COSYVOICE_DIR / "third_party" / "Matcha-TTS"
    assert env.MODELS_DIR == project_root / "models"
    assert env.OUT_DIR == project_root / "out"
    assert cosyvoice_dir == env.COSYVOICE_DIR
    assert matcha_dir == env.MATCHA_TTS_DIR


def test_real_module_paths_are_absolute_and_anchored_off_package_location():
    assert cosyvoice_env.PROJECT_ROOT.is_absolute()
    assert cosyvoice_env.COSYVOICE_DIR == cosyvoice_env.VENDOR_ROOT / "CosyVoice"
    assert (
        cosyvoice_env.MATCHA_TTS_DIR
        == cosyvoice_env.COSYVOICE_DIR / "third_party" / "Matcha-TTS"
    )


def test_add_cosyvoice_to_syspath_raises_when_vendor_dir_missing(fake_project):
    env, _project_root, _cosyvoice_dir, _matcha_dir = fake_project

    with pytest.raises(RuntimeError, match="make vendor"):
        env.add_cosyvoice_to_syspath()


def test_add_cosyvoice_to_syspath_raises_when_matcha_dir_missing(fake_project):
    env, _project_root, cosyvoice_dir, matcha_dir = fake_project
    cosyvoice_dir.mkdir(parents=True)

    assert not matcha_dir.is_dir()
    with pytest.raises(RuntimeError, match="make vendor"):
        env.add_cosyvoice_to_syspath()


def test_add_cosyvoice_to_syspath_succeeds_when_both_dirs_present(fake_project):
    env, _project_root, cosyvoice_dir, matcha_dir = fake_project
    matcha_dir.mkdir(parents=True)

    env.add_cosyvoice_to_syspath()

    assert str(cosyvoice_dir) in sys.path
    assert str(matcha_dir) in sys.path


def test_add_cosyvoice_to_syspath_is_idempotent(fake_project):
    env, _project_root, cosyvoice_dir, matcha_dir = fake_project
    matcha_dir.mkdir(parents=True)

    env.add_cosyvoice_to_syspath()
    env.add_cosyvoice_to_syspath()

    assert sys.path.count(str(cosyvoice_dir)) == 1
    assert sys.path.count(str(matcha_dir)) == 1
