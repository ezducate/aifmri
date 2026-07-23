"""Temporal recording and the BOLD-style timeline."""

import pytest


def test_text_recording_reveals_tokens_progressively(client, transformer):
    r = client.post(f"/api/models/{transformer}/record",
                    data={"mode": "text", "frames": 12,
                          "text": "attention builds across the sentence"})
    assert r.status_code == 200
    d = r.json()
    assert d["frames"] == 12
    assert len(d["labels"]) == 12
    assert len(d["global"]) == 12


def test_recording_captures_attention_per_frame(client, transformer):
    d = client.post(f"/api/models/{transformer}/record",
                    data={"mode": "text", "frames": 10,
                          "text": "the matrix grows"}).json()
    frames = d["attention"]["frames"]["attn"]
    assert len(frames) == 10
    # the attention matrix grows as more tokens are revealed
    assert len(frames[-1]) > len(frames[0])


def test_noise_walk_is_smooth(client, cnn):
    import numpy as np
    d = client.post(f"/api/models/{cnn}/record",
                    data={"mode": "noise", "frames": 24, "seed": 3}).json()
    jumps = np.abs(np.diff(d["global"]))
    assert jumps.max() < 0.5, "slerp walk should not teleport between frames"


def test_seek_reruns_a_frame_at_full_fidelity(client, transformer):
    client.post(f"/api/models/{transformer}/record",
                data={"mode": "text", "frames": 8, "text": "seek me"})
    r = client.post(f"/api/models/{transformer}/record/seek", data={"frame": 4})
    assert r.status_code == 200
    assert "activations" in r.json()


def test_seek_without_recording_is_rejected(client):
    # a FRESH model: the shared fixtures get recorded on by other tests
    mid = client.post("/api/samples/demo-cnn").json()["model_id"]
    r = client.post(f"/api/models/{mid}/record/seek", data={"frame": 0})
    assert r.status_code == 409
