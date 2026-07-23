"""Smoke: everything loads and every endpoint answers."""

import pytest

SAMPLES = ["demo-cnn", "demo-transformer", "demo-audio", "demo-video", "demo-clip"]


def test_index_serves(client):
    assert client.get("/").status_code == 200


def test_sample_list(client):
    names = [s["name"] for s in client.get("/api/samples").json()["samples"]]
    assert set(names) == set(SAMPLES)


@pytest.mark.parametrize("name", SAMPLES)
def test_every_sample_loads_and_runs(client, name):
    d = client.post(f"/api/samples/{name}").json()
    assert d["status"] == "ok"
    assert len(d["graph"]["nodes"]) > 0
    assert d["primary_input"]


def test_hf_gallery_lists_models(client):
    r = client.get("/api/hf")
    assert r.status_code == 200
    assert len(r.json()["models"]) >= 5


def test_modalities_are_detected(client):
    expected = {"demo-cnn": "image", "demo-transformer": "text",
                "demo-audio": "audio", "demo-video": "video"}
    for name, modality in expected.items():
        d = client.post(f"/api/samples/{name}").json()
        prim = next(i for i in d["inputs"] if i["name"] == d["primary_input"])
        assert prim["modality"] == modality, f"{name}: {prim['modality']}"


def test_stats_and_raw_reads(client, cnn):
    assert client.get(f"/api/models/{cnn}/stats",
                      params={"node": "relu1"}).status_code == 200
    assert client.get(f"/api/models/{cnn}/raw",
                      params={"node": "relu1", "limit": 16}).status_code == 200


def test_stimulus_viewer_reports_inputs(client, cnn):
    d = client.get(f"/api/models/{cnn}/stimulus").json()
    assert d["inputs"]
    assert d["inputs"][0]["modality"] == "image"


def test_loaded_models_are_listable(client, cnn):
    assert any(m["model_id"] == cnn
               for m in client.get("/api/models").json()["models"])


def test_hf_gallery_reports_whether_it_can_actually_export(client):
    """So the UI can say so before you click, rather than every button
    failing with a 422 that renders somewhere you cannot see."""
    d = client.get("/api/hf").json()
    assert "available" in d and isinstance(d["available"], bool)
    if not d["available"]:
        assert "pip install" in d["hint"]


def test_version_endpoint(client):
    from app.version import __version__
    assert client.get("/api/version").json()["version"] == __version__
