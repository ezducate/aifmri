"""Robustness guards, each pinning a bug that actually shipped once."""

import numpy as np
import onnx
import pytest
from onnx import helper as h, TensorProto as T


def _load(client, model, name="t.onnx"):
    return client.post("/api/models",
                       files={"file": (name, model.SerializeToString())}).json()


def test_zero_size_intermediates_do_not_crash(client):
    """THE causal-LM bug. GPT-style exports legitimately produce EMPTY
    intermediate tensors (an empty KV cache on the first step). AIFmri called
    numpy reductions on them and threw 'zero-size array to reduction
    operation minimum'. The model was never at fault — our stats code was.
    """
    nodes = [
        # a legitimately empty tensor: slice [0:0] of the sequence axis
        h.make_node("Slice", ["x", "s0", "e0", "ax"], ["empty"], name="empty_cut"),
        h.make_node("Identity", ["empty"], ["empty_out"], name="passthrough"),
        h.make_node("Identity", ["x"], ["y"], name="real"),
    ]
    inits = [
        h.make_tensor("s0", T.INT64, [1], [0]),
        h.make_tensor("e0", T.INT64, [1], [0]),      # end == start -> length 0
        h.make_tensor("ax", T.INT64, [1], [1]),
    ]
    g = h.make_graph(
        nodes, "zero_size",
        [h.make_tensor_value_info("x", T.FLOAT, [1, 4])],
        [h.make_tensor_value_info("y", T.FLOAT, [1, 4]),
         h.make_tensor_value_info("empty_out", T.FLOAT, [1, 0])],
        inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 10

    d = _load(client, m, "empty.onnx")
    assert d.get("status") == "ok"
    r = client.post(f"/api/models/{d['model_id']}/run", data={"mode": "noise"})
    assert r.status_code == 200, r.json()
    # the empty tensor is skipped, the real one is reported
    acts = r.json()["activations"]
    assert "empty_out" not in acts
    assert "y" in acts


def test_non_finite_activations_are_sanitised(client):
    """Untrained/random models overflow to inf, which is not JSON-encodable
    and used to blow up the whole response."""
    big = np.full((4, 4), 1e30, np.float32)
    nodes = [h.make_node("MatMul", ["x", "W"], ["huge"], name="blow_up"),
             h.make_node("MatMul", ["huge", "W"], ["y"], name="blow_up2")]
    inits = [h.make_tensor("W", T.FLOAT, [4, 4], big.ravel())]
    g = h.make_graph(nodes, "overflow",
                     [h.make_tensor_value_info("x", T.FLOAT, [1, 4])],
                     [h.make_tensor_value_info("y", T.FLOAT, [1, 4])], inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 10
    d = _load(client, m, "inf.onnx")
    assert d.get("status") == "ok"
    r = client.post(f"/api/models/{d['model_id']}/run", data={"mode": "noise"})
    assert r.status_code == 200
    import json
    json.dumps(r.json())              # would raise on inf/nan
    for a in r.json()["activations"].values():
        assert all(np.isfinite(v) for v in a["stats"].values()
                   if isinstance(v, float))


def test_unknown_model_id_is_404(client):
    assert client.get("/api/models/nope/weights", params={"node": "x"}
                      ).status_code == 404


def test_unknown_node_is_handled(client, cnn):
    r = client.get(f"/api/models/{cnn}/stats", params={"node": "not_a_layer"})
    assert r.status_code in (404, 409, 422)


@pytest.fixture
def isolated_registry():
    """Eviction is global and destructive (it drops the session), so these
    tests must not touch the session-scoped fixture models — hide them."""
    import app.core as core
    saved = dict(core.MODELS)
    core.MODELS.clear()
    try:
        yield core
    finally:
        core.MODELS.clear()
        core.MODELS.update(saved)


def test_models_are_evicted_to_stay_within_a_memory_budget(
        client, monkeypatch, isolated_registry):
    """Regression: nothing evicted loaded models, so loading several large
    ones OOM-killed the server. Found by the slow suite loading DistilGPT-2
    three times."""
    core = isolated_registry
    monkeypatch.setattr(core, "MODELS_BUDGET_MB", 0.05)   # force pressure
    ids = [client.post("/api/samples/demo-cnn").json()["model_id"]
           for _ in range(4)]
    assert ids[-1] in core.MODELS            # the newest always survives
    assert sum(i in core.MODELS for i in ids) < len(ids)   # older ones dropped


def test_evicted_model_gives_a_clear_error(client, monkeypatch,
                                           isolated_registry):
    core = isolated_registry
    monkeypatch.setattr(core, "MODELS_BUDGET_MB", 0.05)
    first = client.post("/api/samples/demo-cnn").json()["model_id"]
    for _ in range(3):
        client.post("/api/samples/demo-cnn")
    assert first not in core.MODELS
    r = client.get(f"/api/models/{first}/stats", params={"node": "relu1"})
    assert r.status_code == 404
    assert "evicted" in r.json()["detail"].lower()


def test_eviction_never_drops_the_only_model(client, monkeypatch,
                                             isolated_registry):
    core = isolated_registry
    monkeypatch.setattr(core, "MODELS_BUDGET_MB", 0.0001)
    mid = client.post("/api/samples/demo-cnn").json()["model_id"]
    assert mid in core.MODELS
