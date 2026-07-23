"""Jacobian lens — the ONNX finite-difference adaptation of jacobian-lens.

The core claim is falsifiable: transported through the model's own head, a
mid-depth layer should read out what the model actually predicts.
"""

import numpy as np
import pytest


def test_lens_readout_matches_the_models_real_prediction(client, cnn):
    """The lens says what the layer is 'disposed to say'. At mid-depth on a
    settled model that should be the prediction it actually makes."""
    import app.core as core
    lm = core.MODELS[cnn]
    logits = np.asarray(lm.last_activations["logits"], np.float32).reshape(-1)
    predicted = lm.labels[int(np.argmax(logits))]

    r = client.get(f"/api/models/{cnn}/jlens",
                   params={"node": "relu2", "k": 3, "n_probe": 24})
    assert r.status_code == 200
    top = [t["label"] for t in r.json()["topk"]]
    assert predicted in top, f"lens said {top}, model predicts {predicted}"


def test_lens_probabilities_are_normalised(client, cnn):
    d = client.get(f"/api/models/{cnn}/jlens",
                   params={"node": "relu1", "k": 10, "n_probe": 16}).json()
    probs = [t["prob"] for t in d["topk"]]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert probs == sorted(probs, reverse=True)


def test_lens_refuses_without_a_decodable_head(client):
    """A model whose output is a feature vector has nothing to decode into —
    it must say so rather than print vocabulary nonsense."""
    mid = client.post("/api/samples/demo-cnn").json()["model_id"]
    client.post(f"/api/models/{mid}/run", data={"mode": "noise"})
    r = client.get(f"/api/models/{mid}/jlens", params={"node": "relu1"})
    assert r.status_code == 422
    assert "feature" in r.json()["detail"].lower()


def test_lens_works_once_labels_are_attached(client, cnn):
    r = client.get(f"/api/models/{cnn}/jlens",
                   params={"node": "relu1", "k": 3, "n_probe": 16})
    assert r.status_code == 200


def test_lens_needs_a_stimulus(client):
    import json
    mid = client.post("/api/samples/demo-cnn").json()["model_id"]
    client.post(f"/api/models/{mid}/labels",
                files={"file": ("l.json", json.dumps(["a"] * 10).encode())})
    r = client.get(f"/api/models/{mid}/jlens", params={"node": "relu1"})
    assert r.status_code == 409


def test_lens_rejects_layers_wider_than_the_probe_cap(client, cnn):
    r = client.get(f"/api/models/{cnn}/jlens", params={"node": "flat"})
    assert r.status_code == 422
    assert "width" in r.json()["detail"].lower()


def test_stack_view_samples_layers_across_depth(client, cnn):
    r = client.get(f"/api/models/{cnn}/jlens/stack",
                   params={"k": 3, "n_probe": 12, "max_layers": 6})
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert 0 < len(rows) <= 6
    assert [x["depth"] for x in rows] == sorted(x["depth"] for x in rows)


def test_ram_guard_is_measured_not_fixed():
    """Regression: the guard was a fixed 150MB cap; a 3.5x RAM estimate then
    OOM-killed the server. Measured cost is ~5x the model size."""
    import app.core as core
    assert core.SUBGRAPH_RAM_FACTOR >= 4.9
    assert core.JLENS_MAX_HIDDEN >= 768   # DistilGPT-2's hidden width
