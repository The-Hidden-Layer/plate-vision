import os

import pytest

from app.serve_local import configure


def test_host_config_overrides_container_paths_and_rejects_stubs(tmp_path, monkeypatch):
    pytest.importorskip("dotenv")
    monkeypatch.setenv("LPD_WEIGHTS", "/app/weights/lpd.pt")
    monkeypatch.setenv("MEDIA_ROOT", "/app/media")
    monkeypatch.setenv("LPD_BACKEND", "stub")
    monkeypatch.setenv("LPR_BACKEND", "stub")
    config = tmp_path / "serve.env"
    config.write_text(
        "LPD_BACKEND=model\nLPR_BACKEND=model\n"
        "LPD_WEIGHTS=/native/weights/lpd.pt\nMEDIA_ROOT=/native/media\n"
    )
    configure(config)
    assert os.environ["LPD_WEIGHTS"] == "/native/weights/lpd.pt"
    assert os.environ["MEDIA_ROOT"] == "/native/media"
    assert os.environ["LPD_BACKEND"] == "model"
    config.write_text("LPD_BACKEND=stub\nLPR_BACKEND=stub\n")
    with pytest.raises(RuntimeError, match="trained model"):
        configure(config)
    with pytest.raises(RuntimeError, match="missing"):
        configure(tmp_path / "absent.env")
