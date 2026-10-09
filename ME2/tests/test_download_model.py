import pytest

import me2_voicegen.generation.cosyvoice_env as cosyvoice_env
import me2_voicegen.generation.download_model as download_model


def make_complete_manifest(model_dir):
    model_dir.mkdir(parents=True, exist_ok=True)
    for entry in download_model.MANIFEST:
        if entry == "CosyVoice-BlankEN":
            blank_en = model_dir / entry
            blank_en.mkdir(exist_ok=True)
            (blank_en / "model.safetensors").write_bytes(b"fake")
        else:
            (model_dir / entry).write_bytes(b"fake")


def test_target_dir_uses_cosyvoice_env_models_dir(fake_project, monkeypatch):
    env, project_root, _cosyvoice_dir, _matcha_dir = fake_project
    monkeypatch.setattr(download_model, "cosyvoice_env", env)

    assert download_model.target_dir() == project_root / "models" / "CosyVoice2-0.5B"


def test_missing_manifest_entries_empty_when_all_present(tmp_path):
    model_dir = tmp_path / "CosyVoice2-0.5B"
    make_complete_manifest(model_dir)

    assert download_model.missing_manifest_entries(model_dir) == []


def test_missing_manifest_entries_reports_missing_files(tmp_path):
    model_dir = tmp_path / "CosyVoice2-0.5B"
    make_complete_manifest(model_dir)
    (model_dir / "llm.pt").unlink()

    assert download_model.missing_manifest_entries(model_dir) == ["llm.pt"]


def test_missing_manifest_entries_reports_missing_blank_en_subdir(tmp_path):
    model_dir = tmp_path / "CosyVoice2-0.5B"
    make_complete_manifest(model_dir)
    (model_dir / "CosyVoice-BlankEN" / "model.safetensors").unlink()
    (model_dir / "CosyVoice-BlankEN").rmdir()

    assert download_model.missing_manifest_entries(model_dir) == ["CosyVoice-BlankEN"]


def test_missing_manifest_entries_treats_empty_blank_en_dir_as_missing(tmp_path):
    model_dir = tmp_path / "CosyVoice2-0.5B"
    make_complete_manifest(model_dir)
    (model_dir / "CosyVoice-BlankEN" / "model.safetensors").unlink()

    assert download_model.missing_manifest_entries(model_dir) == ["CosyVoice-BlankEN"]


def test_missing_manifest_entries_follows_symlinked_files(tmp_path):
    model_dir = tmp_path / "CosyVoice2-0.5B"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    for entry in download_model.MANIFEST:
        if entry == "CosyVoice-BlankEN":
            real_blank_en = cache_dir / "CosyVoice-BlankEN"
            real_blank_en.mkdir()
            (real_blank_en / "model.safetensors").write_bytes(b"fake")
            (model_dir / entry).symlink_to(real_blank_en, target_is_directory=True)
        else:
            real_file = cache_dir / entry
            real_file.write_bytes(b"fake")
            (model_dir / entry).symlink_to(real_file)

    assert download_model.missing_manifest_entries(model_dir) == []


def test_parse_args_defaults_to_modelscope():
    args = download_model.parse_args([])
    assert args.source == "modelscope"
    assert args.verbose is False


def test_parse_args_accepts_huggingface_source():
    args = download_model.parse_args(["--source", "huggingface"])
    assert args.source == "huggingface"


def test_parse_args_rejects_unknown_source():
    with pytest.raises(SystemExit):
        download_model.parse_args(["--source", "bogus"])


def test_main_skips_download_when_manifest_already_complete(fake_project, monkeypatch):
    env, _project_root, _cosyvoice_dir, _matcha_dir = fake_project
    monkeypatch.setattr(download_model, "cosyvoice_env", env)
    make_complete_manifest(download_model.target_dir())

    called = []
    monkeypatch.setattr(download_model, "download_modelscope", lambda d: called.append(d))
    monkeypatch.setattr(download_model, "download_huggingface", lambda d: called.append(d))

    assert download_model.main([]) == 0
    assert called == []


def test_main_calls_modelscope_downloader_by_default(fake_project, monkeypatch):
    env, _project_root, _cosyvoice_dir, _matcha_dir = fake_project
    monkeypatch.setattr(download_model, "cosyvoice_env", env)

    def fake_download(model_dir):
        make_complete_manifest(model_dir)

    called = []
    monkeypatch.setattr(
        download_model,
        "download_modelscope",
        lambda d: (called.append(("modelscope", d)), fake_download(d)),
    )
    monkeypatch.setattr(
        download_model, "download_huggingface", lambda d: called.append(("huggingface", d))
    )

    assert download_model.main([]) == 0
    assert called == [("modelscope", download_model.target_dir())]


def test_main_calls_huggingface_downloader_when_source_flag_set(fake_project, monkeypatch):
    env, _project_root, _cosyvoice_dir, _matcha_dir = fake_project
    monkeypatch.setattr(download_model, "cosyvoice_env", env)

    def fake_download(model_dir):
        make_complete_manifest(model_dir)

    called = []
    monkeypatch.setattr(
        download_model, "download_modelscope", lambda d: called.append(("modelscope", d))
    )
    monkeypatch.setattr(
        download_model,
        "download_huggingface",
        lambda d: (called.append(("huggingface", d)), fake_download(d)),
    )

    assert download_model.main(["--source", "huggingface"]) == 0
    assert called == [("huggingface", download_model.target_dir())]


def test_main_raises_when_download_leaves_manifest_incomplete(fake_project, monkeypatch):
    env, _project_root, _cosyvoice_dir, _matcha_dir = fake_project
    monkeypatch.setattr(download_model, "cosyvoice_env", env)

    def incomplete_download(model_dir):
        make_complete_manifest(model_dir)
        (model_dir / "llm.pt").unlink()

    monkeypatch.setattr(download_model, "download_modelscope", incomplete_download)

    with pytest.raises(RuntimeError, match="llm.pt"):
        download_model.main([])
