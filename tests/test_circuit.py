"""Circuit tracing: causal influence measured by perturbation."""

import pytest


def test_targets_are_only_downstream_layers(client, cnn):
    d = client.get(f"/api/models/{cnn}/circuit/targets",
                   params={"source": "relu1"}).json()
    assert d["targets"]
    assert all(t["depth"] > d["source_depth"] for t in d["targets"])


def test_boosting_a_source_moves_the_target(client, cnn):
    d = client.get(f"/api/models/{cnn}/circuit/trace",
                   params={"source": "relu1", "target": "pool2",
                           "source_channel": 0, "mode": "boost",
                           "strength": 3}).json()
    assert d["total_change"] > 0
    assert len(d["top"]) > 0


def test_trace_refuses_an_upstream_target(client, cnn):
    r = client.get(f"/api/models/{cnn}/circuit/trace",
                   params={"source": "pool2", "target": "relu1"})
    assert r.status_code == 422
    assert "deeper" in r.json()["detail"].lower()


def test_ablating_a_layer_changes_the_output(client, cnn):
    d = client.get(f"/api/models/{cnn}/circuit/ablate",
                   params={"source": "relu1"}).json()
    assert d["l2_change"] > 0
    assert "pred_changed" in d
    assert d["pred_before"]["label"].startswith("digit_")


def test_ablating_one_channel_hurts_less_than_the_whole_layer(client, cnn):
    whole = client.get(f"/api/models/{cnn}/circuit/ablate",
                       params={"source": "relu1"}).json()["l2_change"]
    one = client.get(f"/api/models/{cnn}/circuit/ablate",
                     params={"source": "relu1", "source_channel": 0}
                     ).json()["l2_change"]
    assert one < whole
