"""Attribution: occlusion, ranking, and activation maximization.

The centrepiece is the matched-filter cross-validation. Two features that share
no code must agree: the maximizer is an evolutionary search that never sees the
weights, the weight viewer reads them straight off disk. Matched-filter theory
says the input that maximally excites a first-layer conv filter IS that filter's
pattern — so they must land on the same filter.

That test is also a warning about measuring the wrong quantity: scored by
correlating the AVERAGE patch it passes only 1/4, because the mean|act|
objective lets the optimum flip sign position-to-position and averaging cancels
exactly the pattern being looked for. Scored sign-invariantly it is 5/5.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------- occlusion

def test_occlusion_peaks_on_the_bright_square(client, cnn, square_image):
    """Ground truth: the only structure in the image is a square in the
    top-left quadrant, so importance must peak there."""
    client.post(f"/api/models/{cnn}/run",
                data={"mode": "image", "normalize": "unit"},
                files={"image": ("x.png", square_image)})
    r = client.get(f"/api/models/{cnn}/attribution/occlusion",
                   params={"node": "relu1", "channel": 0, "grid": 8})
    assert r.status_code == 200
    sal = np.array(r.json()["saliency"])
    assert sal.shape == (8, 8)
    peak_y, peak_x = np.unravel_index(np.argmax(sal), sal.shape)
    # square spans pixels 4..14 of 32 -> grid cells 1..3 of 8
    assert peak_y <= 3 and peak_x <= 3, f"peak at {(peak_y, peak_x)}, expected top-left"


def test_occlusion_on_text_returns_per_token_saliency(client, transformer):
    r = client.get(f"/api/models/{transformer}/attribution/occlusion",
                   params={"node": "attn", "channel": 0})
    assert r.status_code == 200
    d = r.json()
    assert d["kind"] == "tokens"
    assert len(d["saliency"]) == len(d["tokens"])


def test_occlusion_needs_a_stimulus_first(client):
    mid = client.post("/api/samples/demo-cnn").json()["model_id"]
    r = client.get(f"/api/models/{mid}/attribution/occlusion",
                   params={"node": "relu1"})
    assert r.status_code == 409


# ------------------------------------------------------------------ ranking

def test_noise_ranking_is_sorted_descending(client, cnn):
    r = client.get(f"/api/models/{cnn}/attribution/rank_noise",
                   params={"node": "relu1", "channel": 0, "n": 12, "top": 5})
    assert r.status_code == 200
    resp = [x["response"] for x in r.json()["ranked"]]
    assert resp == sorted(resp, reverse=True)


def test_frame_ranking_requires_a_recording(client, cnn):
    r = client.get(f"/api/models/{cnn}/attribution/rank_frames",
                   params={"node": "relu1"})
    assert r.status_code == 409


def test_frame_ranking_after_recording(client, transformer):
    client.post(f"/api/models/{transformer}/record",
                data={"mode": "text", "frames": 8, "text": "ranking frames now"})
    r = client.get(f"/api/models/{transformer}/attribution/rank_frames",
                   params={"node": "attn", "channel": 0, "top": 4})
    assert r.status_code == 200
    d = r.json()
    assert len(d["series"]) == 8
    assert len(d["ranked"]) <= 4


# ------------------------------------------------------- maximization basics

@pytest.fixture(scope="module")
def maximized(client, cnn):
    r = client.post(f"/api/models/{cnn}/attribution/maximize",
                    data={"node": "relu2", "channel": 3, "steps": 120})
    assert r.status_code == 200
    return r.json()


def test_maximization_substantially_increases_the_response(maximized):
    # tuned defaults reach ~16x; anything under 3x means the search regressed
    assert maximized["gain"] > 3.0
    assert maximized["best"] > maximized["start"]


def test_maximization_trace_best_never_decreases(maximized):
    t = maximized["trace"]
    running = np.maximum.accumulate(t)
    assert maximized["best"] >= running[-1] - 1e-6


def test_maximization_returns_a_repaintable_run(maximized):
    """The synthesized input is run server-side so the whole UI can repaint."""
    assert "run" in maximized and "activations" in maximized["run"]


def test_maximized_input_is_viewable(client, cnn, maximized):
    r = client.get(f"/api/models/{cnn}/stimulus/image", params={"input": "input"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"


def test_different_channels_want_different_inputs(client, cnn):
    import io
    from PIL import Image
    seen = {}
    for ch in (0, 7):
        client.post(f"/api/models/{cnn}/attribution/maximize",
                    data={"node": "relu2", "channel": ch, "steps": 80})
        png = client.get(f"/api/models/{cnn}/stimulus/image",
                         params={"input": "input"}).content
        seen[ch] = np.asarray(Image.open(io.BytesIO(png)).convert("L"), float)
    assert np.abs(seen[0] - seen[7]).mean() > 2.0


def test_maximize_rejects_unknown_layer(client, cnn):
    r = client.post(f"/api/models/{cnn}/attribution/maximize",
                    data={"node": "does_not_exist"})
    assert r.status_code == 422


def test_maximize_on_token_inputs_uses_discrete_search(client, transformer):
    r = client.post(f"/api/models/{transformer}/attribution/maximize",
                    data={"node": "attn", "channel": 1, "steps": 40})
    assert r.status_code == 200
    d = r.json()
    assert d["kind"] == "tokens"
    assert len(d["tokens"]) > 0


# ------------------------------------------- matched-filter cross-validation

def _mean_abs_patch_corr(X: np.ndarray, w: np.ndarray) -> float:
    """Mean |correlation| between each 3x3x3 patch of X and kernel w.

    Sign-invariant ON PURPOSE: the objective is mean|activation|, so the
    optimum may flip sign from position to position. Correlating the mean
    patch instead cancels those and scores ~1/4 where this scores 5/5.
    """
    patches = np.stack([X[:, i:i + 3, j:j + 3].ravel()
                        for i in range(0, 29, 2) for j in range(0, 29, 2)])
    patches = patches - patches.mean(axis=1, keepdims=True)
    pn = np.linalg.norm(patches, axis=1) + 1e-8
    wc = w.ravel() - w.mean()
    return float(np.mean(np.abs(patches @ wc) / (pn * (np.linalg.norm(wc) + 1e-8))))


@pytest.mark.parametrize("k", [0, 3, 7, 11, 14])
def test_maximizer_rediscovers_the_actual_conv_filter(client, cnn,
                                                      conv1_kernels, k):
    """The maximizer never sees the weights; it must still land on filter k."""
    import app.core as core
    r = client.post(f"/api/models/{cnn}/attribution/maximize",
                    data={"node": "conv1", "channel": k, "steps": 200,
                          "regularize": "false"})
    assert r.status_code == 200
    X = np.asarray(core.MODELS[cnn].last_feed["input"], np.float32)[0]
    cors = [_mean_abs_patch_corr(X, conv1_kernels[f])
            for f in range(len(conv1_kernels))]
    best = int(np.argmax(cors))
    others = np.mean([c for f, c in enumerate(cors) if f != k])
    assert best == k, f"maximizing conv1[{k}] matched filter {best} instead"
    assert cors[k] / others > 1.2
