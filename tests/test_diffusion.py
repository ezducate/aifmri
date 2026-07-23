"""Diffusion mode: a denoising UNet's sequence axis is NOISE LEVEL.

The temporal engine already recorded activations over a stimulus sequence; a
denoising loop is exactly that, so the BOLD carpet becomes "which layers do the
work at which noise level".

Most of these build a tiny UNet-shaped ONNX by hand so they run in
milliseconds. The real-model proof (a trained DDPM actually turning noise into
a photograph) lives in test_gallery_slow.py.
"""

import numpy as np
import pytest
from onnx import helper as h, TensorProto as T

import app.core as core


def _toy_unet(channels=4, hw=8):
    """(sample, timestep) -> out_sample. The minimum shape of a denoiser:
    something image-like to clean up, and a rank-0 timestep knob."""
    rng = np.random.default_rng(0)
    nodes = [
        h.make_node("Unsqueeze", ["timestep", "ax"], ["t1"], name="t_expand"),
        h.make_node("Mul", ["sample", "k"], ["scaled"], name="block1"),
        h.make_node("Conv", ["scaled", "w"], ["conv"], name="block2",
                    kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        h.make_node("Tanh", ["conv"], ["out_sample"], name="head"),
    ]
    inits = [
        h.make_tensor("ax", T.INT64, [1], [0]),
        h.make_tensor("k", T.FLOAT, [1], [0.01]),
        h.make_tensor("w", T.FLOAT, [channels, channels, 3, 3],
                      (rng.standard_normal((channels, channels, 3, 3)) * .2)
                      .astype(np.float32).ravel()),
    ]
    g = h.make_graph(
        nodes, "toy_unet",
        [h.make_tensor_value_info("sample", T.FLOAT, [1, channels, hw, hw]),
         h.make_tensor_value_info("timestep", T.FLOAT, [])],
        [h.make_tensor_value_info("out_sample", T.FLOAT, [1, channels, hw, hw])],
        inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 10
    return m


@pytest.fixture(scope="module")
def unet(client):
    d = client.post("/api/models",
                    files={"file": ("unet.onnx", _toy_unet().SerializeToString())}
                    ).json()
    assert d["status"] == "ok"
    return d


def test_rank_zero_timestep_does_not_crash_the_loader(unet):
    """Regression: a rank-0 scalar input threw IndexError in modality
    detection — nothing else in the zoo has one."""
    ts = next(i for i in unet["inputs"] if i["name"] == "timestep")
    assert ts["shape"] == []
    assert ts["modality"] == "scalar"


def test_latent_is_detected_and_chosen_as_the_stimulus(unet):
    sample = next(i for i in unet["inputs"] if i["name"] == "sample")
    assert sample["modality"] == "latent"
    assert unet["primary_input"] == "sample", "a UNet is stimulated through " \
                                              "its latent, not its timestep"


def test_latent_spatial_dims_resolve_small_not_224():
    """Regression: resolving a latent like an image (224) made the UNet's
    self-attention ask onnxruntime for ~5 GB."""
    assert core.resolve_dims([1, 4, "height", "width"], "sample") == \
        [1, 4, core.LATENT_HW, core.LATENT_HW]
    # a real image input must still resolve to 224
    assert core.resolve_dims([1, 3, "height", "width"], "pixel_values") == \
        [1, 3, 224, 224]


def test_model_is_reported_as_diffusion(unet):
    assert unet["diffusion"] is True


def test_a_classifier_is_not_diffusion(client, cnn):
    d = client.post("/api/samples/demo-cnn").json()
    assert d["diffusion"] is False


def test_denoise_recording_walks_the_noise_schedule(client, unet):
    mid = unet["model_id"]
    r = client.post(f"/api/models/{mid}/record",
                    data={"mode": "denoise", "frames": 8, "schedule": "linear"})
    assert r.status_code == 200
    d = r.json()
    assert d["frames"] == 8
    # labels are timesteps, descending from full noise to clean
    ts = [int(l.split("=")[1]) for l in d["labels"]]
    assert ts[0] == 999 and ts[-1] == 0
    assert ts == sorted(ts, reverse=True)


def test_denoise_is_refused_on_a_non_diffusion_model(client, cnn):
    r = client.post(f"/api/models/{cnn}/record",
                    data={"mode": "denoise", "frames": 4})
    assert r.status_code == 422
    assert "timestep" in r.json()["detail"].lower()


def test_seek_replays_a_denoising_step(client, unet):
    mid = unet["model_id"]
    client.post(f"/api/models/{mid}/record",
                data={"mode": "denoise", "frames": 6, "schedule": "linear"})
    assert client.post(f"/api/models/{mid}/record/seek",
                       data={"frame": 3}).status_code == 200


@pytest.mark.parametrize("schedule", ["scaled_linear", "linear", "cosine"])
def test_beta_schedules_are_valid_and_distinct(schedule):
    ac = core._alphas_cumprod(1000, schedule)
    assert ac.shape == (1000,)
    assert np.all(ac > 0) and np.all(ac <= 1.0)
    assert np.all(np.diff(ac) <= 1e-9), "alpha_cumprod must be non-increasing"


def test_schedules_actually_differ():
    a = core._alphas_cumprod(1000, "linear")
    b = core._alphas_cumprod(1000, "scaled_linear")
    assert np.abs(a - b).max() > 0.05, "picking the schedule must matter"


def test_trajectory_stays_bounded_on_a_garbage_model(client, unet):
    """A wrong noise prediction gets amplified ~15x per step at t=999, so an
    untrained UNet diverges to inf without the clip_sample-style clamp."""
    import app.core as core
    lm = core.MODELS[unet["model_id"]]
    prim = next(i for i in lm.inputs if i["name"] == lm.primary_input)
    lats, ts = core._ddim_trajectory(lm, prim, 12, seed=1, schedule="linear")
    assert all(np.isfinite(x).all() for x in lats)
    assert max(float(np.abs(x).max()) for x in lats) < 1e4
