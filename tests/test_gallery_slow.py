"""Real HuggingFace models. Skipped by default — they download hundreds of MB.

    pytest -m slow

These are the tests that prove the tool works on architectures people actually
use, rather than on toy graphs we generated ourselves.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def _total_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return 0.0


#: DistilGPT-2 is ~2.4 GB resident once the exposed graph and session are up.
#: Anything that then allocates on top of it needs real headroom.
needs_ram = pytest.mark.skipif(
    0 < _total_ram_gb() < 8,
    reason=f"needs >=8 GB RAM (this box has {_total_ram_gb():.1f} GB)")


@pytest.fixture(scope="module")
def optimum():
    pytest.importorskip("optimum.exporters.onnx",
                        reason="needs optimum + optimum-onnx")


def test_bert_tiny_has_real_wordpiece_attention(client, optimum):
    d = client.post("/api/hf/bert-tiny").json()
    assert d.get("tokenizer_attached") is True
    mid = d["model_id"]
    r = client.post(f"/api/models/{mid}/run",
                    data={"mode": "text", "text": "the quick brown fox"})
    assert r.status_code == 200
    toks = [t["token"] for t in r.json()["tokens"]]
    assert toks[0] == "[CLS]" and "quick" in toks


def test_vit_gives_patch_attention(client, optimum):
    import io
    from PIL import Image
    d = client.post("/api/hf/vit-tiny").json()
    mid = d["model_id"]
    buf = io.BytesIO()
    Image.effect_noise((224, 224), 90).convert("RGB").save(buf, "PNG")
    client.post(f"/api/models/{mid}/run",
                data={"mode": "image", "normalize": "imagenet"},
                files={"image": ("x.png", buf.getvalue())})
    attn = [n["id"] for n in d["graph"]["nodes"] if n.get("attention")]
    assert attn
    a = client.get(f"/api/models/{mid}/attention",
                   params={"node": attn[0]}).json()
    assert a["q"] == a["k"] == 197        # 196 patches + CLS


@pytest.fixture(scope="module")
def gpt(client, optimum):
    """One DistilGPT-2 for the whole module — it is ~480 MB resident, and
    loading a copy per test is exactly what exhausted memory before."""
    d = client.post("/api/hf/distilgpt2").json()
    assert d.get("status") == "ok"
    r = client.post(f"/api/models/{d['model_id']}/run",
                    data={"mode": "text", "text": "The capital of France is"})
    assert r.status_code == 200, "causal-LM zero-size guard regressed"
    return d


def test_distilgpt2_attention_is_strictly_causal(client, gpt):
    """The single best validation the attention extractor has: nothing in the
    code assumes causality, so a perfectly lower-triangular matrix can only
    come from the model itself."""
    d = gpt
    mid = d["model_id"]
    attn = [n["id"] for n in d["graph"]["nodes"] if n.get("attention")]
    assert len(attn) >= 6
    a = client.get(f"/api/models/{mid}/attention",
                   params={"node": attn[0]}).json()
    M = np.array(a["matrix"])
    assert np.triu(M, 1).sum() < 1e-6, "a token attended to the future"
    assert np.allclose(M.sum(axis=1), 1.0, atol=0.02)


def test_big_models_refuse_the_lens_cleanly_instead_of_ooming(client, gpt):
    """Never let a RAM estimate OOM-kill the server: refuse with a reason."""
    d = gpt
    mid = d["model_id"]
    node = next(n["id"] for n in d["graph"]["nodes"]
                if n.get("attention"))
    r = client.get(f"/api/models/{mid}/jlens", params={"node": node})
    # either it fits on this machine, or it says exactly why it doesn't
    assert r.status_code in (200, 422)
    if r.status_code == 422:
        assert "MB" in r.json()["detail"]


@needs_ram
def test_everything_else_works_on_a_big_model(client, gpt):
    mid = gpt["model_id"]
    assert client.get(f"/api/models/{mid}/health",
                      params={"n": 4}).status_code == 200
    assert client.get(f"/api/models/{mid}/latency",
                      params={"runs": 2}).status_code == 200
    assert client.post(f"/api/models/{mid}/export",
                       data={"notes": "big"}).status_code == 200


@pytest.fixture(scope="module")
def ddpm(tmp_path_factory):
    """A REAL trained diffusion model (google/ddpm-cifar10-32), exported
    straight from diffusers. ~143 MB of weights, pixel-space, unconditional."""
    pytest.importorskip("diffusers")
    torch = pytest.importorskip("torch")
    from diffusers import UNet2DModel
    import app.core as core

    out = tmp_path_factory.mktemp("ddpm") / "unet.onnx"
    unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").eval()
    n, c = unet.config.sample_size, unet.config.in_channels

    class Wrap(torch.nn.Module):
        def __init__(s, m):
            super().__init__()
            s.m = m

        def forward(s, sample, timestep):
            return s.m(sample, timestep).sample

    torch.onnx.export(Wrap(unet), (torch.randn(1, c, n, n), torch.tensor(999.0)),
                      str(out), input_names=["sample", "timestep"],
                      output_names=["out_sample"], opset_version=17)
    core.load_model(out, source_name="ddpm-cifar10")
    return core.MODELS[list(core.MODELS)[-1]]


def _neighbour_corr(img: np.ndarray) -> float:
    """~0 for iid noise, ~0.9 for a photograph."""
    a = img[0].mean(axis=0)
    return float(np.corrcoef(a[:, :-1].ravel(), a[:, 1:].ravel())[0, 1])


def test_ddim_loop_turns_pure_noise_into_an_image(ddpm):
    """The whole claim of diffusion mode in one assertion: run the real
    scheduler on a real trained model and structure must emerge."""
    import app.core as core
    prim = next(i for i in ddpm.inputs if i["name"] == ddpm.primary_input)
    lats, ts = core._ddim_trajectory(ddpm, prim, 40, seed=7, schedule="linear")
    assert ts[0] == 999 and ts[-1] == 0
    assert abs(_neighbour_corr(lats[0])) < 0.2, "step 0 should be iid noise"
    assert _neighbour_corr(lats[-1]) > 0.6, "no image emerged from the loop"
    assert float(np.abs(lats[-1]).max()) < 1.5, "a DDPM sample lives in [-1,1]"


def test_the_noise_schedule_choice_matters(ddpm):
    """It is not in the ONNX file, so it has to be a user choice: the wrong
    schedule produces a plausible-looking but under-denoised trajectory."""
    import app.core as core
    prim = next(i for i in ddpm.inputs if i["name"] == ddpm.primary_input)
    right, _ = core._ddim_trajectory(ddpm, prim, 40, seed=7, schedule="linear")
    wrong, _ = core._ddim_trajectory(ddpm, prim, 40, seed=7,
                                     schedule="scaled_linear")
    assert _neighbour_corr(right[-1]) > _neighbour_corr(wrong[-1]) + 0.3


def test_denoising_carpet_records_every_layer(client, ddpm):
    import app.core as core
    rec = core.start_recording(ddpm, "denoise", 16, schedule="linear")
    assert rec["frames"] == 16
    assert len(rec["timeline"]) > 300
    g = np.array(rec["global"])
    assert np.isfinite(g).all()
    peaks = {n: int(np.argmax(s)) for n, s in rec["timeline"].items()
             if max(s) > 1e-6}
    # different layers do their work at different noise levels
    assert len(set(peaks.values())) > 3
