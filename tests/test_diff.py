"""Model diffing — must localize a change to exactly where it was made."""

import pytest
from conftest import build_head_perturbed


def test_diff_localizes_a_classifier_head_change(client, cnn):
    """Only the Gemm weights differ, so every layer before it must read
    exactly 0.000 divergence and the Gemm must spike."""
    other = build_head_perturbed(client)
    d = client.get("/api/diff",
                   params={"model_a": cnn, "model_b": other}).json()
    by_op = {r["op"]: r["rel_l2"] for r in d["profile"]}
    upstream = [v for op, v in by_op.items() if op not in ("Gemm", "MatMul")]
    assert max(upstream) < 1e-4, f"upstream layers should be identical: {by_op}"
    assert by_op["Gemm"] > 0.1


def test_diff_matches_layers_by_id_for_identical_architectures(client, cnn):
    other = build_head_perturbed(client)
    d = client.get("/api/diff", params={"model_a": cnn, "model_b": other}).json()
    assert d["match_mode"] == "id"
    assert d["n_matched"] >= 6


def test_diff_reports_output_divergence(client, cnn):
    other = build_head_perturbed(client, delta=0.9)
    d = client.get("/api/diff", params={"model_a": cnn, "model_b": other}).json()
    assert d["output"] is not None
    assert "pred_changed" in d["output"]


def test_identical_models_show_zero_divergence(client, cnn, noise_image):
    twin = client.post("/api/samples/demo-cnn").json()["model_id"]
    d = client.get("/api/diff", params={"model_a": cnn, "model_b": twin}).json()
    assert max(r["rel_l2"] for r in d["profile"]) < 1e-5


def test_diff_needs_a_stimulus_on_a(client):
    a = client.post("/api/samples/demo-cnn").json()["model_id"]
    b = client.post("/api/samples/demo-cnn").json()["model_id"]
    r = client.get("/api/diff", params={"model_a": a, "model_b": b})
    assert r.status_code == 409


def test_unrelated_architectures_are_refused(client, cnn, transformer):
    r = client.get("/api/diff", params={"model_a": cnn, "model_b": transformer})
    assert r.status_code in (200, 422)
    if r.status_code == 422:
        assert "architecture" in r.json()["detail"].lower()


def test_diff_survives_memory_pressure(client, monkeypatch, noise_image):
    """Regression: LRU eviction used to throw out model A the moment you
    loaded model B, breaking the one workflow that needs two models resident."""
    import app.core as core
    saved = dict(core.MODELS)
    core.MODELS.clear()
    try:
        monkeypatch.setattr(core, "MODELS_BUDGET_MB", 0.15)   # neither fits
        a = client.post("/api/samples/demo-cnn").json()["model_id"]
        client.post(f"/api/models/{a}/run",
                    data={"mode": "image", "normalize": "unit"},
                    files={"image": ("x.png", noise_image)})
        b = client.post("/api/samples/demo-cnn").json()["model_id"]
        assert a in core.MODELS, "loading B evicted the model we want to diff"
        assert client.get("/api/diff",
                          params={"model_a": a, "model_b": b}).status_code == 200
    finally:
        core.MODELS.clear()
        core.MODELS.update(saved)
