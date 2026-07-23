"""Weight viewer, latency lens, and session export."""

import pytest


# ------------------------------------------------------------------ weights

def test_conv_weights_are_reported(client, cnn):
    d = client.get(f"/api/models/{cnn}/weights", params={"node": "conv1"}).json()
    assert d["has_weights"] is True
    kinds = {t["kind"] for t in d["tensors"]}
    assert "kernel" in kinds and "bias" in kinds


def test_weight_stats_are_finite_and_sane(client, cnn):
    d = client.get(f"/api/models/{cnn}/weights", params={"node": "conv1"}).json()
    for t in d["tensors"]:
        assert t["min"] <= t["mean"] <= t["max"]
        assert t["l2"] >= 0
        assert 0 <= t["zeros_pct"] <= 100
        assert len(t["hist"]) > 0


def test_first_conv_kernels_are_viewable(client, cnn):
    d = client.get(f"/api/models/{cnn}/weights", params={"node": "conv1"}).json()
    assert d["viewable_kernels"] is True
    r = client.get(f"/api/models/{cnn}/weights/image", params={"node": "conv1"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 200


def test_weightless_layers_say_so(client, cnn):
    d = client.get(f"/api/models/{cnn}/weights", params={"node": "relu1"}).json()
    assert d["has_weights"] is False


# ------------------------------------------------------------------ latency

def test_latency_profile_is_internally_consistent(client, cnn):
    d = client.get(f"/api/models/{cnn}/latency", params={"runs": 6}).json()
    assert d["runs"] >= 1
    assert d["total_us"] > 0
    assert all(p["us"] >= 0 for p in d["profile"])
    node_sum = sum(p["us"] for p in d["profile"])
    # everything the profiler attributed must add up to the reported total
    assert abs(node_sum - d["total_us"]) <= max(50, 0.05 * d["total_us"])


def test_latency_percentages_are_sane(client, cnn):
    d = client.get(f"/api/models/{cnn}/latency", params={"runs": 4}).json()
    assert all(0 <= t["pct"] <= 100 for t in d["top"])


def test_latency_is_honest_about_operator_fusion(client, cnn):
    """ONNX Runtime folds Relu into Conv. Reporting 'Relu = 41% of runtime'
    without saying so would be a lie, so fusion must be surfaced."""
    d = client.get(f"/api/models/{cnn}/latency", params={"runs": 4}).json()
    assert "fused" in d
    assert "overhead_us" in d


def test_latency_top_is_sorted(client, cnn):
    d = client.get(f"/api/models/{cnn}/latency", params={"runs": 4}).json()
    us = [t["us"] for t in d["top"]]
    assert us == sorted(us, reverse=True)


# ------------------------------------------------------------------- export

def test_export_is_self_contained(client, cnn):
    r = client.post(f"/api/models/{cnn}/export", data={"notes": "unit test"})
    assert r.status_code == 200
    html = r.content.decode()
    assert "data:image" in html            # images inlined, not linked
    assert "<img src=\"http" not in html    # nothing fetched from the network


def test_export_includes_the_users_notes(client, cnn):
    note = "a distinctive phrase for the report"
    html = client.post(f"/api/models/{cnn}/export",
                       data={"notes": note}).content.decode()
    assert note in html


def test_export_optional_sections(client, cnn):
    html = client.post(f"/api/models/{cnn}/export",
                       data={"notes": "x", "include_latency": "true"}
                       ).content.decode()
    assert "Latency" in html
