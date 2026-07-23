"""One-click HuggingFace model gallery.

Each entry is exported to ONNX on first request via optimum, cached on disk,
and comes with the metadata AIFmri needs (task, tokenizer, default stimulus,
normalization). Kept to small/tiny checkpoints so first-load stays quick.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# repo, optimum task, and UI hints. `trained` marks genuinely-trained
# checkpoints (real attention patterns) vs tiny-random test models (fast,
# structure-only).
HF_MODELS = {
    "bert-tiny": {
        "title": "BERT (tiny) — text encoder",
        "repo": "prajjwal1/bert-tiny",
        "task": "feature-extraction",
        "modality": "text",
        "desc": "A real 2-layer BERT. Type a sentence and watch self-attention "
                "across wordpieces.",
        "default_text": "the quick brown fox jumps over the lazy dog",
        "trained": True,
    },
    "distilbert-sst2": {
        "title": "DistilBERT — sentiment",
        "repo": "distilbert-base-uncased-finetuned-sst-2-english",
        "task": "text-classification",
        "modality": "text",
        "desc": "6-layer DistilBERT fine-tuned on sentiment. Attention here "
                "actually tracks meaning.",
        "default_text": "this movie was absolutely wonderful and moving",
        "trained": True,
    },
    "vit-tiny": {
        "title": "ViT — image transformer",
        "repo": "WinKawaks/vit-tiny-patch16-224",
        "task": "image-classification",
        "modality": "image",
        "desc": "Vision Transformer. Feed an image (or webcam) and see patch "
                "attention. Uses ImageNet normalization.",
        "normalize": "imagenet",
        "trained": True,
    },
    "whisper-tiny": {
        "title": "Whisper (tiny) — speech encoder",
        "repo": "openai/whisper-tiny",
        "task": "automatic-speech-recognition",
        "modality": "audio",
        "desc": "Whisper's encoder. Upload speech or record from the mic; "
                "attention runs over the mel frames.",
        "encoder_only": True,
        "trained": True,
    },
    "ddpm-cifar10": {
        "title": "DDPM (CIFAR-10) — diffusion",
        "repo": "google/ddpm-cifar10-32",
        "task": "unconditional-image-generation",
        "modality": "image",
        "desc": "A real trained denoising UNet. Record the DENOISING LOOP and "
                "watch pure noise become an image — the carpet's time axis "
                "becomes noise level. Use the 'linear' schedule.",
        "diffusion": True,
        "schedule": "linear",
        "trained": True,
    },
    "distilgpt2": {
        "title": "DistilGPT-2 — causal LM",
        "repo": "distilgpt2",
        "task": "text-generation",
        "modality": "text",
        "desc": "A real generative GPT. Reveal a sentence token by token "
                "(temporal mode) to watch causal attention build. ~480 MB — "
                "first load takes a minute.",
        "default_text": "The capital of France is",
        "trained": True,
    },
}


def _export_diffusion_unet(repo: str, outdir: Path) -> None:
    """optimum's exporter assumes a text encoder, which unconditional
    diffusion pipelines don't have — so export the UNet ourselves.

    The result takes (sample, timestep) and predicts noise, which is exactly
    the signature AIFmri's denoise recorder drives.
    """
    import torch
    from diffusers import UNet2DModel

    unet = UNet2DModel.from_pretrained(repo).eval()
    n, c = unet.config.sample_size, unet.config.in_channels

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, sample, timestep):
            return self.m(sample, timestep).sample

    torch.onnx.export(
        _Wrap(unet),
        (torch.randn(1, c, n, n), torch.tensor(999.0)),
        str(outdir / "model.onnx"),
        input_names=["sample", "timestep"], output_names=["out_sample"],
        opset_version=17)


def _pick_onnx(outdir: Path) -> Path:
    """Encoder model if present (Whisper), else the main model.onnx."""
    prefer = ["encoder_model.onnx", "model.onnx", "decoder_model.onnx"]
    for name in prefer:
        p = outdir / name
        if p.exists():
            return p
    onnxs = list(outdir.glob("*.onnx"))
    if not onnxs:
        raise FileNotFoundError("Export produced no .onnx file.")
    return onnxs[0]


def export_hf(name: str, cache_root: Path) -> dict:
    """Export (or reuse cached) a gallery model. Returns paths + metadata."""
    if name not in HF_MODELS:
        raise KeyError(f"No gallery model '{name}'.")
    meta = HF_MODELS[name]
    outdir = cache_root / f"hf_{name}"
    marker = outdir / ".exported"
    if not marker.exists():
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            if meta.get("diffusion"):
                _export_diffusion_unet(meta["repo"], outdir)
                marker.write_text("ok")
                onnx_path = _pick_onnx(outdir)
                return {"onnx_path": onnx_path, "tokenizer_path": None,
                        "meta": meta}
            from optimum.exporters.onnx import main_export
            main_export(meta["repo"], output=str(outdir), task=meta["task"],
                        opset=17, monolith=False)
        except Exception as e:
            shutil.rmtree(outdir, ignore_errors=True)
            raise RuntimeError(
                f"Could not export {meta['repo']} — {type(e).__name__}: "
                f"{str(e)[:180]}") from e
        marker.write_text("ok")

    onnx_path = _pick_onnx(outdir)
    tok = outdir / "tokenizer.json"
    return {
        "onnx_path": onnx_path,
        "tokenizer_path": tok if tok.exists() else None,
        "meta": meta,
    }


def exporter_available() -> tuple[bool, str]:
    """Can this interpreter actually export a gallery model?

    Checked up front so the gallery can say so before you click, instead of
    every button failing with a 422 that renders somewhere you cannot see.
    """
    import importlib.util as u
    missing = [m for m in ("transformers", "optimum") if u.find_spec(m) is None]
    if u.find_spec("optimum") is not None and \
            u.find_spec("optimum.exporters.onnx") is None:
        missing.append("optimum-onnx")
    if missing:
        return False, "pip install " + " ".join(missing) + " onnxscript"
    return True, ""


def gallery_list() -> list[dict]:
    return [{"name": k, "title": v["title"], "desc": v["desc"],
             "modality": v["modality"], "trained": v.get("trained", False),
             "diffusion": v.get("diffusion", False)}
            for k, v in HF_MODELS.items()]
