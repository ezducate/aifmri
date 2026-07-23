"""AIFmri core v0.2 — ingestion, shape resolution, tokenization, activations.

New in v0.2:
  * Multi-input models (input_ids + attention_mask + token_type_ids, etc.)
    with dtype-aware auto-fill of companion inputs.
  * Automatic symbolic-dimension resolution (batch -> 1, seq -> 32,
    image-like -> 224) validated by a probe run at load time.
  * Text stimuli via HF `tokenizers` (upload a tokenizer.json) or a built-in
    byte-level tokenizer that works with any vocab >= 256.
  * Bare state_dict recovery: fingerprint key names/shapes against the
    torchvision zoo, or rebuild from user-pasted model code.
"""

from __future__ import annotations

import io
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
from onnx import shape_inference

MAX_UNITS = 1024
SPATIAL_MAX = 64
RAW_SLICE_MAX = 4096
DEFAULT_SEQ = 32

ORT_DTYPES = {
    "tensor(float)": np.float32, "tensor(float16)": np.float16,
    "tensor(double)": np.float64, "tensor(int64)": np.int64,
    "tensor(int32)": np.int32, "tensor(bool)": np.bool_,
    "tensor(uint8)": np.uint8, "tensor(int8)": np.int8,
}

# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------

class ByteTokenizer:
    """UTF-8 bytes as token ids — works with any model whose vocab >= 256."""
    kind = "byte"

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def id_to_token(self, i: int) -> str:
        try:
            ch = bytes([i]).decode("utf-8")
            return ch if ch.isprintable() else repr(ch)
        except Exception:
            return f"<byte {i}>"

    def vocab_size(self) -> int:
        return 256


class HFTokenizer:
    kind = "hf"

    def __init__(self, tokenizer_json: bytes):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_str(tokenizer_json.decode("utf-8"))

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def id_to_token(self, i: int) -> str:
        t = self.tok.id_to_token(i)
        return t if t is not None else f"#{i}"

    def vocab_size(self) -> int:
        try:
            return int(self.tok.get_vocab_size())
        except Exception:
            return 256


# ---------------------------------------------------------------------------
# Shape resolution
# ---------------------------------------------------------------------------

_SEQ_HINTS = ("seq", "len", "token", "time", "text", "position", "ctx")
_BATCH_HINTS = ("batch", "n", "b")

#: Stable Diffusion works in a 64x64x4 latent. Anything larger explodes the
#: UNet's self-attention (a 224x224 latent asks onnxruntime for ~5 GB).
LATENT_HW = 64


def _looks_latent(name: str, shape: list) -> bool:
    n = (name or "").lower()
    if not any(w in n for w in ("latent", "sample", "noise")):
        return False
    if len(shape) != 4:
        return False
    c = shape[1]
    return (not isinstance(c, int)) or c in (4, 8, 9, 16)


def resolve_dims(shape: list, name: str = "", ndim_hint: int | None = None,
                 seq: int = DEFAULT_SEQ) -> list[int]:
    """Replace symbolic / unknown dims with sensible concrete values."""
    out = []
    for i, d in enumerate(shape):
        if isinstance(d, int) and d > 0:
            out.append(d)
            continue
        label = (d if isinstance(d, str) else "").lower()
        if i == 0 or any(h == label or h in label for h in _BATCH_HINTS if len(h) > 1) \
                or "batch" in label:
            out.append(1)
        elif any(h in label for h in _SEQ_HINTS):
            out.append(seq)
        elif len(shape) == 4 and i >= 2:          # spatial dims
            # A diffusion latent is 64x64 for SD; resolving it like an image
            # (224) makes self-attention allocate tens of GB.
            out.append(LATENT_HW if _looks_latent(name, shape) else 224)
        elif len(shape) == 4 and i == 1:          # channels
            out.append(4 if _looks_latent(name, shape) else 3)
        elif len(shape) == 2 and i == 1:          # (batch, seq) style
            out.append(seq)
        else:
            out.append(seq if i == len(shape) - 1 and len(shape) <= 3 else 1)
    return out


def companion_fill(name: str, shape: list[int], dtype) -> np.ndarray:
    """Auto-fill secondary inputs by name convention."""
    n = name.lower()
    if "mask" in n:
        return np.ones(shape, dtype=dtype)
    if "type" in n or "segment" in n:
        return np.zeros(shape, dtype=dtype)
    if "position" in n:
        return np.arange(shape[-1], dtype=dtype).reshape(
            [1] * (len(shape) - 1) + [shape[-1]]).repeat(shape[0], 0) \
            if len(shape) >= 1 else np.zeros(shape, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def _export_torch(model, input_shape: list[int] | None, out: Path) -> Path:
    import torch
    model.eval()
    shape = input_shape or [1, 3, 224, 224]
    torch.onnx.export(model, torch.randn(*shape), str(out), opset_version=18,
                      input_names=["input"], output_names=["output"])
    return out


def fingerprint_state_dict(sd: dict) -> list[dict]:
    """Rank torchvision architectures by state_dict key/shape overlap."""
    try:
        import torchvision.models as tvm
    except ImportError:
        return []
    keys = set(sd.keys())
    prefixes = {k.split(".")[0] for k in keys}
    # cheap family shortlist by signature prefixes
    families = {
        "resnet": ("conv1", "layer1"), "vgg": ("features", "classifier"),
        "densenet": ("features",), "mobilenet": ("features",),
        "efficientnet": ("features",), "squeezenet": ("features",),
        "shufflenet": ("conv1", "stage2"), "regnet": ("stem", "trunk_output"),
        "convnext": ("features",), "vit": ("conv_proj", "encoder"),
        "swin": ("features",), "alexnet": ("features", "classifier"),
        "googlenet": ("conv1", "inception3a"), "inception": ("Conv2d_1a_3x3",),
        "mnasnet": ("layers",), "maxvit": ("stem", "blocks"),
    }
    shortlist = [m for m in tvm.list_models()
                 if any(f in m for f, sig in families.items()
                        if all(p in prefixes for p in sig))]
    # exclude families that can't be cleanly rebuilt/exported or that
    # download backbone weights during construction
    _skip = ("quantized_", "fcn", "deeplab", "lraspp", "retinanet",
             "fasterrcnn", "maskrcnn", "keypointrcnn", "ssd", "raft",
             "fcos", "r3d", "mc3", "r2plus1d", "s3d", "mvit", "swin3d")
    shortlist = [m for m in shortlist
                 if not any(m.startswith(s) or s in m for s in _skip)]
    if not shortlist:
        shortlist = [m for m in tvm.list_models()[:60]
                     if not any(s in m for s in _skip)]
    scored = []
    for name in shortlist:
        try:
            try:
                ref = tvm.get_model(name, weights=None,
                                    weights_backbone=None).state_dict()
            except TypeError:
                ref = tvm.get_model(name, weights=None).state_dict()
        except Exception:
            continue
        rk = set(ref.keys())
        inter = keys & rk
        if not inter:
            continue
        key_score = len(inter) / len(keys | rk)
        shape_hits = sum(1 for k in inter
                         if tuple(ref[k].shape) == tuple(sd[k].shape))
        shape_score = shape_hits / max(1, len(inter))
        score = 0.5 * key_score + 0.5 * shape_score
        if score > 0.35:
            scored.append({"arch": name, "score": round(score, 3),
                           "matched_keys": len(inter),
                           "exact_shapes": shape_hits})
    scored.sort(key=lambda x: -x["score"])
    return scored[:8]


def torch_from_state_dict(sd: dict, arch: str | None, code: str | None,
                          input_shape: list[int] | None, out: Path) -> Path:
    import torch
    if code:
        ns: dict = {"torch": torch, "nn": torch.nn}
        exec(code, ns)                                    # local, user's own code
        model = ns.get("model")
        if model is None and callable(ns.get("build")):
            model = ns["build"]()
        if model is None:
            for v in ns.values():
                if isinstance(v, torch.nn.Module):
                    model = v
                    break
        if model is None:
            raise ValueError("Code must define `model = ...`, a `build()` "
                             "function, or instantiate an nn.Module.")
    elif arch:
        import torchvision.models as tvm
        model = tvm.get_model(arch, weights=None)
    else:
        raise ValueError("Provide an architecture name or model code.")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > len(sd) * 0.5:
        raise ValueError(f"Architecture mismatch: {len(missing)} of "
                         f"{len(sd)} weight tensors did not fit.")
    return _export_torch(model, input_shape, out)


def _torch_to_onnx(path: Path, input_shape) -> Path | dict:
    import torch
    out = path.with_suffix(".converted.onnx")
    try:
        model = torch.jit.load(str(path), map_location="cpu")
        return _export_torch(model, input_shape, out)
    except Exception:
        pass
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        return _export_torch(obj, input_shape, out)
    if isinstance(obj, dict):
        sd = obj.get("state_dict", obj.get("model_state_dict", obj))
        if isinstance(sd, dict) and sd and \
                all(hasattr(v, "shape") for v in sd.values()):
            return {"needs_arch": True, "state_dict_path": str(path),
                    "candidates": fingerprint_state_dict(sd),
                    "n_tensors": len(sd)}
    raise ValueError("Unrecognized .pt payload — expected TorchScript, an "
                     "nn.Module, or a state_dict.")


def _tf_to_onnx(path: Path) -> Path:
    import tf2onnx
    import tensorflow as tf
    out = path.with_suffix(".converted.onnx")
    if path.suffix in (".h5", ".keras"):
        model = tf.keras.models.load_model(str(path))
        spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input"),)
        proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec,
                                              opset=17)
        onnx.save(proto, str(out))
    else:
        proto, _ = tf2onnx.convert.from_saved_model(str(path), opset=17)
        onnx.save(proto, str(out))
    return out


def to_onnx(path: Path, input_shape=None):
    suffix = path.suffix.lower()
    if suffix == ".onnx":
        return path
    if suffix in (".pt", ".pth"):
        try:
            return _torch_to_onnx(path, input_shape)
        except ImportError:
            raise ValueError(
                "PyTorch is not installed on this server. Either "
                "`pip install torch torchvision`, or export your model to "
                "ONNX first: torch.onnx.export(model, dummy, 'model.onnx').")
    if suffix in (".h5", ".keras", ".pb") or path.is_dir():
        try:
            return _tf_to_onnx(path)
        except ImportError:
            raise ValueError(
                "TensorFlow/tf2onnx are not installed on this server. Either "
                "`pip install tensorflow tf2onnx`, or convert to ONNX first: "
                "python -m tf2onnx.convert --saved-model dir --output m.onnx.")
    raise ValueError(f"Unsupported model format: '{suffix}'. "
                     "Supported: .onnx, .pt, .pth, .h5, .keras, SavedModel.")


# ---------------------------------------------------------------------------
# Modality detection + media decoding
# ---------------------------------------------------------------------------

_MELISH = (32, 40, 64, 80, 96, 128)


#: Diffusion UNets take a rank-0 (or 1-element) timestep. Nothing else in the
#: zoo does, and a scalar has no spatial/temporal structure to stimulate — it
#: is a knob, so it gets its own modality.
_SCALARISH = ("timestep", "time_step", "t_emb", "sigma", "guidance", "scale")


def detect_modality(inp: dict) -> str:
    """Infer what kind of stimulus an input expects from its name/shape/dtype."""
    n = inp["name"].lower()
    shape = inp["shape"]
    rank = len(shape)
    if rank == 0 or (rank == 1 and shape and shape[0] == 1
                     and any(w in n for w in _SCALARISH)):
        return "scalar"                                  # diffusion timestep
    if any(w in n for w in _SCALARISH) and rank <= 1:
        return "scalar"
    # a diffusion latent: (B, 4, H, W) — 4 channels, not 1/3 like an image
    if (rank == 4 and shape[1] in (4, 8, 9, 16)
            and any(w in n for w in ("latent", "sample", "noise"))):
        return "latent"
    if any(w in n for w in ("input_ids", "token", "text")):
        return "text"
    if any(w in n for w in ("pixel", "image", "img")):
        return "video" if rank == 5 else "image"
    if any(w in n for w in ("audio", "wav", "speech", "mel", "spectrogram")):
        return "audio"
    if any(w in n for w in ("video", "clip", "frames")):
        return "video"
    if inp["dtype"].startswith("int") and rank <= 2:
        return "text"
    if rank == 5:
        return "video"
    if rank == 4 and shape[1] in (1, 3):
        # (1,1,mel,frames) is audio-shaped; (1,1,H,W) grayscale is image-shaped
        if shape[1] == 1 and shape[2] in _MELISH and shape[3] > shape[2] * 2:
            return "audio"
        return "image"
    if rank == 4 and shape[3] in (1, 3):
        return "image"                                   # NHWC
    if (rank == 3 and (shape[1] in _MELISH or shape[2] in _MELISH)
            and not any(w in n for w in ("hidden", "encoder", "embed",
                                         "cond", "context"))):
        return "audio"                                   # (1, mel, frames)
    if rank <= 2 and shape[-1] >= 4000:
        return "audio"                                   # raw waveform
    return "tensor"


def decode_audio(data: bytes, target_sr: int = 16000) -> np.ndarray:
    """Decode wav/flac/ogg/mp3 to mono float32 at target_sr (linear resample)."""
    import soundfile as sf
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    y = y.mean(axis=1)                                   # mono
    if sr != target_sr and len(y) > 1:
        t_old = np.linspace(0, 1, len(y), endpoint=False)
        t_new = np.linspace(0, 1, int(len(y) * target_sr / sr), endpoint=False)
        y = np.interp(t_new, t_old, y).astype(np.float32)
    peak = np.abs(y).max()
    return (y / peak if peak > 0 else y).astype(np.float32)


def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    pts = mel2hz(np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for m in range(1, n_mels + 1):
        a, b, c = bins[m - 1], bins[m], bins[m + 1]
        for k in range(a, b):
            if b > a:
                fb[m - 1, k] = (k - a) / (b - a)
        for k in range(b, c):
            if c > b:
                fb[m - 1, k] = (c - k) / (c - b)
    return fb


def log_mel(y: np.ndarray, sr: int, n_mels: int, n_frames: int,
            n_fft: int = 400, hop: int = 160) -> np.ndarray:
    """(n_mels, n_frames) log-mel spectrogram, padded/trimmed to n_frames."""
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    win = np.hanning(n_fft).astype(np.float32)
    n = 1 + (len(y) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        y, shape=(n, n_fft),
        strides=(y.strides[0] * hop, y.strides[0])) * win
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2          # (n, bins)
    mel = spec @ _mel_filterbank(n_mels, n_fft, sr).T        # (n, mels)
    mel = np.log10(np.maximum(mel, 1e-10)).T.astype(np.float32)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    if n_frames > 0:
        if mel.shape[1] < n_frames:
            mel = np.pad(mel, ((0, 0), (0, n_frames - mel.shape[1])))
        mel = mel[:, :n_frames]
    return mel


def audio_to_tensor(data: bytes, shape: list[int], raw_shape: list,
                    sr: int = 16000) -> np.ndarray:
    """Fit a decoded waveform into whatever audio layout the model wants."""
    return fit_audio(decode_audio(data, sr), shape, raw_shape, sr)


def fit_audio(y: np.ndarray, shape: list[int], raw_shape: list,
              sr: int = 16000) -> np.ndarray:
    rank = len(shape)
    if rank <= 2 or (rank == 3 and shape[1] == 1):       # raw waveform
        # (L,) / (1, L) / (1, 1, L) — mel models never have a singleton dim-1
        li = rank - 1
        L = raw_shape[li] if li < len(raw_shape) and \
            isinstance(raw_shape[li], int) and raw_shape[li] > 0 else 0
        if L:                                            # fixed length
            y = np.pad(y, (0, max(0, L - len(y))))[:L]
        return y.reshape([1] * (rank - 1) + [len(y)]).astype(np.float32) \
            if rank > 1 else y.astype(np.float32)
    if rank == 3:                                        # (1, mel, frames) or (1, frames, mel)
        mel_axis = 1 if shape[1] in _MELISH else 2
        n_mels = shape[mel_axis]
        n_frames = shape[3 - mel_axis]
        m = log_mel(y, sr, n_mels, n_frames)
        if mel_axis == 2:
            m = m.T
        return m[None, ...].astype(np.float32)
    if rank == 4:                                        # (1, 1, mel, frames)
        m = log_mel(y, sr, shape[2], shape[3])
        return m[None, None, ...].astype(np.float32)
    raise ValueError(f"Don't know how to fit audio into a rank-{rank} input.")


def video_to_tensor(data: bytes, shape: list[int]) -> np.ndarray:
    """Decode a video, sample frames evenly, fit to NCTHW / NTCHW / NTHWC."""
    import tempfile as _tf

    import imageio.v3 as iio
    with _tf.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(data)
        tmp = f.name
    frames = np.stack(list(iio.imiter(tmp, plugin="FFMPEG")))    # (F, H, W, 3)
    if len(shape) != 5:
        raise ValueError("Video stimulus needs a rank-5 input tensor.")
    dims = shape[1:]                                     # 4 dims after batch
    c_axis = next((i for i, d in enumerate(dims) if d in (1, 3)), 0)
    layouts = {0: "NCTHW", 2: "NTHWC" if dims[-1] in (1, 3) else "NTCHW",
               1: "NTCHW", 3: "NTHWC"}
    layout = layouts.get(c_axis, "NCTHW")
    if layout == "NCTHW":
        C, T, H, W = dims
    elif layout == "NTCHW":
        T, C, H, W = dims
    else:
        T, H, W, C = dims
    idx = np.linspace(0, len(frames) - 1, T).astype(int)
    from PIL import Image
    sel = np.stack([np.asarray(Image.fromarray(frames[i]).resize((W, H)))
                    for i in idx]).astype(np.float32) / 255.0   # (T,H,W,3)
    if C == 1:
        sel = sel.mean(axis=3, keepdims=True)
    if layout == "NCTHW":
        out = sel.transpose(3, 0, 1, 2)
    elif layout == "NTCHW":
        out = sel.transpose(0, 3, 1, 2)
    else:
        out = sel
    return out[None, ...].astype(np.float32)




@dataclass
class LoadedModel:
    model_id: str
    onnx_path: Path
    graph: dict
    session: object = None
    output_names: list[str] = field(default_factory=list)
    inputs: list[dict] = field(default_factory=list)   # name/shape/dtype/raw
    primary_input: str = ""
    last_activations: dict = field(default_factory=dict)
    labels: list[str] | None = None
    tokenizer: object = None
    last_tokens: list[dict] | None = None
    source_name: str = ""
    recording: dict | None = None
    last_stimulus: dict | None = None      # raw stimulus for the inputs viewer
    last_feed: dict | None = None          # actual input tensors of last run
    jlens: dict | None = None              # cached averaged Jacobian per layer
    _proto_cache: object = None            # loaded ONNX proto for subgraph reuse
    last_used: float = 0.0                 # for LRU eviction


def _human_shape(vi) -> list:
    dims = []
    for d in vi.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.dim_value > 0 else (d.dim_param or -1))
    return dims


def _expose_all_intermediates(model: onnx.ModelProto) -> onnx.ModelProto:
    model = shape_inference.infer_shapes(model)
    inits = {i.name for i in model.graph.initializer}
    seen = {o.name for o in model.graph.output}
    for node in model.graph.node:
        for out in node.output:
            if out and out not in seen and out not in inits:
                model.graph.output.append(
                    onnx.helper.make_empty_tensor_value_info(out))
                seen.add(out)
    return model


def build_graph_json(model: onnx.ModelProto) -> dict:
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    shape_by_name = {vi.name: _human_shape(vi)
                     for vi in list(g.value_info) + list(g.output) + list(g.input)}
    producer = {}
    for idx, node in enumerate(g.node):
        for out in node.output:
            producer[out] = idx
    depths = [0] * len(g.node)
    for idx, node in enumerate(g.node):
        for inp in node.input:
            if inp in producer:
                depths[idx] = max(depths[idx], depths[producer[inp]] + 1)
    lane_counter: dict[int, int] = {}
    nodes, edges = [], []
    # detect attention: a Softmax whose input chain (through Div/Add/Mul scaling)
    # comes from a MatMul — that MatMul is QKᵀ, so the Softmax output is the
    # attention weight matrix [..., query, key].
    out_producer = {o: g.node[producer[o]] for o in producer}

    # fused attention kernels (Flash-Attention-style) never expose the softmax
    # matrix — they compute it in one op. We can flag them but not read the
    # [Q,K] weights, so they're marked opaque.
    _FUSED_ATTN = {
        "Attention", "MultiHeadAttention", "GroupQueryAttention",
        "PagedAttention", "SparseAttention", "DecoderMaskedMultiHeadAttention",
        "ScaledDotProductAttention", "MemoryEfficientAttention",
        "FlashAttention",
    }

    def _is_attention(node) -> bool:
        if node.op_type != "Softmax":
            return False
        cur = node.input[0] if node.input else None
        for _ in range(4):                       # walk back past scale/mask ops
            p = out_producer.get(cur)
            if p is None:
                return False
            if p.op_type == "MatMul":
                return True
            if p.op_type in ("Div", "Mul", "Add", "Sub", "Where", "Cast"):
                cur = p.input[0]
                continue
            return False
        return False

    for idx, node in enumerate(g.node):
        main_out = node.output[0] if node.output else f"node_{idx}"
        params = sum(int(np.prod(inits[i].dims)) if inits[i].dims else 1
                     for i in node.input if i in inits)
        d = depths[idx]
        lane = lane_counter.get(d, 0)
        lane_counter[d] = lane + 1
        rec = {"id": main_out, "name": node.name or f"{node.op_type}_{idx}",
               "op": node.op_type, "shape": shape_by_name.get(main_out, []),
               "params": params, "depth": d, "lane": lane}
        if _is_attention(node):
            rec["attention"] = True
        elif node.op_type in _FUSED_ATTN:
            rec["attention"] = True
            rec["attention_fused"] = True         # opaque: no [Q,K] matrix
        nodes.append(rec)
        for inp in node.input:
            if inp in producer:
                edges.append({"from": g.node[producer[inp]].output[0],
                              "to": main_out})
    return {"nodes": nodes, "edges": edges,
            "outputs": [{"name": o.name, "shape": _human_shape(o)}
                        for o in g.output],
            "depth_max": max(depths) if depths else 0,
            "lanes_max": max(lane_counter.values()) if lane_counter else 1}


# ---------------------------------------------------------------------------
# Registry + loading
# ---------------------------------------------------------------------------

MODELS: dict[str, LoadedModel] = {}

# Every loaded model keeps an onnxruntime session (plus the exposed-graph copy)
# resident. Nothing used to evict them, so loading a few large models in one
# session would OOM the server — the slow tests found this by loading
# DistilGPT-2 three times. Keep a rolling budget and drop the least recently
# used, never the one just loaded or currently in use.
MODELS_BUDGET_MB = 1500
# Never evict the most-recent K models however tight the budget. Diffing needs
# TWO resident at once, and loading the comparison model must not throw out the
# one you wanted to compare it against — that is the whole workflow.
MODELS_MIN_KEEP = 2


def touch_model(model_id: str) -> None:
    lm = MODELS.get(model_id)
    if lm is not None:
        lm.last_used = time.time()


def _model_mb(lm: LoadedModel) -> float:
    try:
        mb = lm.onnx_path.stat().st_size / 1e6
        exposed = lm.onnx_path.with_suffix(".exposed.onnx")
        if exposed.exists():
            mb += exposed.stat().st_size / 1e6
        return mb
    except Exception:
        return 0.0


def evict_models(keep: str | None = None) -> list[str]:
    """Drop least-recently-used models until we are back inside the budget,
    always sparing the MODELS_MIN_KEEP most recent (see above)."""
    dropped = []
    while len(MODELS) > MODELS_MIN_KEEP:
        total = sum(_model_mb(m) for m in MODELS.values())
        if total <= MODELS_BUDGET_MB:
            break
        recent = sorted(MODELS.values(),
                        key=lambda m: getattr(m, "last_used", 0.0),
                        reverse=True)[:MODELS_MIN_KEEP]
        spared = {m.model_id for m in recent} | ({keep} if keep else set())
        victim = min((m for mid, m in MODELS.items() if mid not in spared),
                     key=lambda m: getattr(m, "last_used", 0.0), default=None)
        if victim is None:
            break
        MODELS.pop(victim.model_id, None)
        victim.session = None
        victim._proto_cache = None
        dropped.append(victim.model_id)
    return dropped
PENDING: dict[str, dict] = {}     # state_dict recovery sessions


def _pick_primary(inputs: list[dict]) -> str:
    """The input the user actually stimulates (not masks/types/positions)."""
    def rank(i):
        n = i["name"].lower()
        # a diffusion UNet is stimulated through its latent, not through the
        # text embedding or the timestep knob
        if any(w in n for w in ("latent", "sample")) and len(i["shape"]) == 4:
            return -1
        if any(w in n for w in _SCALARISH):
            return 3
        if any(w in n for w in ("mask", "type", "segment", "position",
                                "encoder_hidden", "hidden_states")):
            return 2
        if "id" in n or "input" in n or i["dtype"].startswith("int"):
            return 0
        return 1
    return sorted(inputs, key=rank)[0]["name"]


def load_model(path: Path, input_shape=None, source_name: str = ""):
    import onnxruntime as ort

    result = to_onnx(path, input_shape)
    if isinstance(result, dict):                    # state_dict — needs arch
        pid = uuid.uuid4().hex[:12]
        PENDING[pid] = {"path": str(path), "input_shape": input_shape,
                        "source_name": source_name}
        result["pending_id"] = pid
        return result

    proto = onnx.load(str(result))
    graph_json = build_graph_json(proto)
    exposed_path = result.with_suffix(".exposed.onnx")
    onnx.save(_expose_all_intermediates(proto), str(exposed_path))
    sess = ort.InferenceSession(str(exposed_path),
                                providers=["CPUExecutionProvider"])

    inputs = []
    for si in sess.get_inputs():
        raw = list(si.shape)
        resolved = resolve_dims(raw, si.name)
        if input_shape and si.name == sess.get_inputs()[0].name:
            resolved = input_shape
        rec = {"name": si.name, "raw_shape": raw, "shape": resolved,
               "dtype": str(np.dtype(ORT_DTYPES.get(si.type, np.float32)))}
        rec["modality"] = detect_modality(rec)
        inputs.append(rec)

    lm = LoadedModel(
        model_id=uuid.uuid4().hex[:12], onnx_path=result, graph=graph_json,
        session=sess, output_names=[o.name for o in sess.get_outputs()],
        inputs=inputs, primary_input=_pick_primary(inputs),
        source_name=source_name or path.name)

    probe = _probe(lm)                              # validate resolved shapes
    lm.last_used = time.time()
    MODELS[lm.model_id] = lm
    evict_models(keep=lm.model_id)
    return {"model": lm, "probe": probe}


def _probe(lm: LoadedModel) -> dict:
    """Zero-stimulus dry run to confirm the resolved shapes actually work."""
    try:
        feed = {}
        for i in lm.inputs:
            dt = np.dtype(i["dtype"])
            feed[i["name"]] = np.zeros(i["shape"], dtype=dt)
        lm.session.run(lm.output_names, feed)
        return {"ok": True}
    except Exception as e:
        return {"ok": False,
                "hint": ("Auto-resolved input shapes failed a probe run — "
                         "edit the shapes in the Inputs table and reload. "
                         f"Runtime said: {str(e)[:300]}")}


def resolve_pending(pending_id: str, arch: str | None, code: str | None,
                    input_shape=None):
    import torch
    info = PENDING.get(pending_id)
    if info is None:
        raise KeyError("Unknown or expired recovery session — re-upload the file.")
    path = Path(info["path"])
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj.get("model_state_dict", obj)) \
        if isinstance(obj, dict) else obj
    out = path.with_suffix(".converted.onnx")
    onnx_path = torch_from_state_dict(sd, arch, code,
                                      input_shape or info.get("input_shape"), out)
    del PENDING[pending_id]
    return load_model(onnx_path, input_shape or info.get("input_shape"),
                      source_name=info.get("source_name", path.name))


def apply_shapes(lm: LoadedModel, shapes: dict[str, list[int]]) -> dict:
    for i in lm.inputs:
        if i["name"] in shapes:
            i["shape"] = [int(v) for v in shapes[i["name"]]]
    return _probe(lm)


# ---------------------------------------------------------------------------
# Stimulus construction
# ---------------------------------------------------------------------------

_COMPANION_WORDS = ("mask", "type", "segment", "position")


def is_companion(name: str) -> bool:
    return any(w in name.lower() for w in _COMPANION_WORDS)


def stimulus_inputs(lm: LoadedModel) -> list[dict]:
    stims = [i for i in lm.inputs if not is_companion(i["name"])]
    return stims or lm.inputs



# Image normalization presets: (mean, std) in 0..1 space, or a callable.
NORM_PRESETS = {
    "unit":     None,                                        # 0..1 as-is
    "signed":   ("_signed", None),                           # -1..1
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "clip":     ([0.48145466, 0.4578275, 0.40821073],
                 [0.26862954, 0.26130258, 0.27577711]),
    "inception":([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),          # == signed
    "openai":   ([0.48145466, 0.4578275, 0.40821073],
                 [0.26862954, 0.26130258, 0.27577711]),      # alias for clip
}


def apply_norm(arr: np.ndarray, normalize: str, c: int) -> np.ndarray:
    """arr is HxWxC in 0..1. Returns normalized array."""
    preset = NORM_PRESETS.get(normalize, None)
    if preset is None:
        return arr
    if preset[0] == "_signed" or normalize == "signed":
        return arr * 2.0 - 1.0
    if c == 3:
        mean = np.array(preset[0], np.float32)
        std = np.array(preset[1], np.float32)
        return (arr - mean) / std
    return arr


def _stim_tensor(lm: LoadedModel, inp: dict, mode: str, blob=None,
                 json_values=None, text=None, seed=None, normalize="unit",
                 sample_rate: int = 16000) -> np.ndarray:
    """Build one input tensor from one stimulus spec."""
    dt = np.dtype(inp["dtype"])
    shape = list(inp["shape"])

    if mode in ("text", "tokens"):
        if mode == "text":
            tok = lm.tokenizer or ByteTokenizer()
            ids = tok.encode(text or "")
            lm.tokenizer = tok
        else:
            ids = [int(v) for v in (json_values or [])]
        if not ids:
            raise ValueError(f"'{inp['name']}': the stimulus text tokenized "
                             "to nothing.")
        seq_axis = len(shape) - 1
        raw = inp["raw_shape"][seq_axis] \
            if seq_axis < len(inp["raw_shape"]) else -1
        if isinstance(raw, int) and raw > 0:
            ids = (ids + [0] * raw)[:raw]
        x = np.asarray(ids, dtype=dt).reshape(
            [1] * (len(shape) - 1) + [len(ids)])
        if lm.tokenizer:
            lm.last_tokens = [{"id": int(i),
                               "token": lm.tokenizer.id_to_token(int(i))}
                              for i in ids]
        return x
    if mode == "noise":
        rng = np.random.default_rng(seed)
        return (rng.standard_normal(shape).astype(np.float32)
                if dt.kind == "f" else
                rng.integers(0, 100, size=shape).astype(dt))
    if mode == "zeros":
        return np.zeros(shape, dtype=dt)
    if mode == "json":
        return np.asarray(json_values, dtype=dt).reshape(shape)
    if mode == "audio":
        if not blob:
            raise ValueError(f"'{inp['name']}': audio stimulus needs an "
                             "audio file (or a mic recording).")
        return audio_to_tensor(blob, shape, inp["raw_shape"], sr=sample_rate)
    if mode == "video":
        if not blob:
            raise ValueError(f"'{inp['name']}': video stimulus needs a "
                             "video file.")
        return video_to_tensor(blob, shape)
    if mode == "image":
        if not blob:
            raise ValueError(f"'{inp['name']}': image stimulus needs an "
                             "image file.")
        from PIL import Image
        img = Image.open(io.BytesIO(blob)).convert("RGB")
        if len(shape) == 4 and shape[1] in (1, 3):
            c, h, w, chw = shape[1], shape[2], shape[3], True
        elif len(shape) == 4:
            h, w, c, chw = shape[1], shape[2], shape[3], False
        else:
            raise ValueError(f"'{inp['name']}': image stimulus needs a 4-D "
                             f"input, got {shape}. Try noise instead.")
        img = img.resize((w, h))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if c == 1:
            arr = arr.mean(axis=2, keepdims=True)
        arr = apply_norm(arr, normalize, c)
        if chw:
            arr = arr.transpose(2, 0, 1)
        return arr[None, ...].astype(np.float32)
    raise ValueError(f"Unknown stimulus mode: {mode}")


def _fill_companions(lm: LoadedModel, feed: dict) -> None:
    """Auto-fill mask/type/position inputs, following the dynamic shape of
    the stimulus input they belong to (a text input if ranks match)."""
    stims = [i for i in stimulus_inputs(lm) if i["name"] in feed]
    for inp in lm.inputs:
        if inp["name"] in feed:
            continue
        sh = list(inp["shape"])
        same_rank = [s for s in stims
                     if feed[s["name"]].ndim == len(sh)]
        pref = [s for s in same_rank if s.get("modality") == "text"] \
            or same_rank
        if pref:
            ref = feed[pref[0]["name"]].shape
            for ax in range(len(sh)):
                r = inp["raw_shape"][ax] if ax < len(inp["raw_shape"]) else -1
                if not (isinstance(r, int) and r > 0):
                    sh[ax] = ref[ax]
        feed[inp["name"]] = companion_fill(inp["name"], sh,
                                           np.dtype(inp["dtype"]))


def make_multi_feed(lm: LoadedModel, specs: dict[str, dict]) -> dict:
    """Build a full feed from per-input stimulus specs.

    specs: {input_name: {mode, text?, values?, seed?, normalize?,
                         sample_rate?, _blob?}}
    Stimulus inputs missing from specs default to zeros.
    """
    lm.last_tokens = None
    feed = {}
    preview = {}
    for inp in stimulus_inputs(lm):
        s = specs.get(inp["name"], {"mode": "zeros"})
        feed[inp["name"]] = _stim_tensor(
            lm, inp, s.get("mode", "zeros"), blob=s.get("_blob"),
            json_values=s.get("values"), text=s.get("text"),
            seed=s.get("seed"), normalize=s.get("normalize", "unit"),
            sample_rate=int(s.get("sample_rate", 16000)))
        # remember what the raw stimulus was, per input, for the viewer
        preview[inp["name"]] = {
            "mode": s.get("mode", "zeros"),
            "modality": inp.get("modality", "tensor"),
            "text": s.get("text"),
            "has_blob": s.get("_blob") is not None,
            "sample_rate": int(s.get("sample_rate", 16000)),
            "_normalize": s.get("normalize", "unit"),
        }
    lm.last_stimulus = preview
    _fill_companions(lm, feed)
    return feed


def make_feed(lm: LoadedModel, mode: str, image_bytes=None, json_values=None,
              text: str | None = None, seed=None, normalize="unit",
              sample_rate: int = 16000) -> dict:
    """Single-stimulus wrapper: the primary input gets the stimulus, other
    stimulus inputs get zeros, companions are auto-filled."""
    return make_multi_feed(lm, {lm.primary_input: {
        "mode": mode, "_blob": image_bytes, "values": json_values,
        "text": text, "seed": seed, "normalize": normalize,
        "sample_rate": sample_rate}})


# ---------------------------------------------------------------------------
# Activation capture + summarisation  (unchanged maths from v0.1)
# ---------------------------------------------------------------------------

def _finite(x):
    """JSON-safe float: map NaN/±inf to 0.0 (and clamp absurd magnitudes)."""
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    v = float(x)
    if not math.isfinite(v):
        return 0.0
    return v


def _sanitize_array(a: np.ndarray) -> np.ndarray:
    """Replace non-finite entries so untrained/overflowing models stay usable."""
    if a.dtype.kind == "f" and not np.isfinite(a).all():
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def _unit_values(arr: np.ndarray):
    a = np.abs(arr.astype(np.float32))
    if a.ndim == 4:
        per, kind = a.mean(axis=(0, 2, 3)), "channel"
    elif a.ndim == 3:
        per, kind = a.mean(axis=(0, 2)), "token" if a.shape[1] <= a.shape[2] else "channel"
        # (B, T, D): per-token signal is usually more interesting for text
        if arr.shape[1] * 4 < arr.shape[2] or True:
            per, kind = a.mean(axis=(0, 2)), "token"
    elif a.ndim == 2:
        per, kind = a.mean(axis=0), "feature"
    else:
        per, kind = a.reshape(-1), "value"
    n = per.shape[0]
    if n > MAX_UNITS:
        per = per[np.linspace(0, n - 1, MAX_UNITS).astype(int)]
    return per.tolist(), int(n), kind


def _downsample2d(m: np.ndarray) -> np.ndarray:
    h, w = m.shape
    sh, sw = max(1, h // SPATIAL_MAX), max(1, w // SPATIAL_MAX)
    if sh > 1 or sw > 1:
        hh, ww = (h // sh) * sh, (w // sw) * sw
        m = m[:hh, :ww].reshape(hh // sh, sh, ww // sw, sw).mean(axis=(1, 3))
    return m


def run_stimulus(lm: LoadedModel, feed: dict) -> dict:
    outs = lm.session.run(lm.output_names, feed)
    acts = {k: (_sanitize_array(v) if isinstance(v, np.ndarray) else v)
            for k, v in zip(lm.output_names, outs)}
    lm.last_activations = acts
    lm.last_feed = {k: v for k, v in feed.items()}   # for the inputs viewer
    per_node = {}
    for node in lm.graph["nodes"]:
        name = node["id"]
        arr = acts.get(name)
        if not isinstance(arr, np.ndarray) or arr.dtype.kind not in "fiu":
            continue
        if arr.size == 0:            # legal but empty (causal-LM exports)
            continue
        arr_f = arr.astype(np.float32)
        absa = np.abs(arr_f)
        units, n_units, kind = _unit_values(arr_f)
        per_node[name] = {
            "stats": {"mean": _finite(arr_f.mean()), "std": _finite(arr_f.std()),
                      "min": _finite(arr_f.min()), "max": _finite(arr_f.max()),
                      "mean_abs": _finite(absa.mean()),
                      "l2": _finite(np.linalg.norm(arr_f) / math.sqrt(arr_f.size)),
                      "sparsity": _finite((absa < 1e-6).mean()),
                      "size": int(arr_f.size), "shape": list(arr_f.shape)},
            "units": [round(_finite(v), 5) for v in units],
            "n_units": n_units, "unit_kind": kind}
    all_units = np.concatenate([np.asarray(v["units"])
                                for v in per_node.values()]) \
        if per_node else np.zeros(1)
    scale = _finite(np.percentile(all_units, 99)) or 1.0
    return {"activations": per_node, "scale": scale, "tokens": lm.last_tokens}


def raw_slice(lm, node, channel, offset, limit):
    if node not in lm.last_activations:
        raise KeyError("No activation recorded for this layer yet — run a "
                       "stimulus first (left panel).")
    a = np.asarray(lm.last_activations[node], dtype=np.float32)
    if channel is not None and a.ndim >= 3:
        a = a[0, channel] if a.ndim == 4 else a[0, :, channel]
    flat = a.reshape(-1)
    limit = min(limit, RAW_SLICE_MAX)
    return {"node": node, "shape": list(a.shape), "offset": offset,
            "total": int(flat.size),
            "values": [_finite(v) for v in flat[offset:offset + limit]]}


def spatial_for(lm, node, channel):
    if node not in lm.last_activations:
        raise KeyError("No activation recorded yet — run a stimulus first.")
    arr = np.asarray(lm.last_activations[node], dtype=np.float32)
    if arr.ndim == 4:
        m = np.abs(arr[0, channel]) if channel is not None \
            else np.abs(arr[0]).mean(axis=0)
    elif arr.ndim == 3:                     # (B, T, D) -> token x dim heatmap
        m = np.abs(arr[0])
    elif arr.ndim == 2:
        side = int(math.ceil(math.sqrt(arr.shape[1])))
        pad = np.zeros(side * side, np.float32)
        pad[:arr.shape[1]] = np.abs(arr[0])
        m = pad.reshape(side, side)
    else:
        return {"map": None, "channel": channel}
    return {"map": np.round(_downsample2d(m), 5).tolist(), "channel": channel}


def decode_as_image(lm, node) -> bytes:
    from PIL import Image
    if node not in lm.last_activations:
        raise KeyError("No activation recorded yet — run a stimulus first.")
    arr = np.asarray(lm.last_activations[node], dtype=np.float32)
    if arr.ndim == 4:
        t = arr[0]
        img = t[:3] if t.shape[0] >= 3 \
            else np.repeat(t.mean(axis=0, keepdims=True), 3, axis=0)
        img = img.transpose(1, 2, 0)
    elif arr.ndim == 3:
        m = np.abs(arr[0])
        img = np.stack([m] * 3, axis=2)
    elif arr.ndim == 2:
        side = int(math.ceil(math.sqrt(arr.shape[1])))
        pad = np.zeros(side * side, np.float32)
        pad[:arr.shape[1]] = arr[0]
        img = np.stack([pad.reshape(side, side)] * 3, axis=2)
    else:
        raise ValueError("This tensor has no 2-D structure to render.")
    lo, hi = img.min(), img.max()
    img = (img - lo) / (hi - lo + 1e-8)
    pil = Image.fromarray((img * 255).astype(np.uint8))
    if max(pil.size) < 256:
        f = max(1, 256 // max(pil.size))
        pil = pil.resize((pil.width * f, pil.height * f), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def decode_topk(lm, node, k=10) -> dict:
    if node not in lm.last_activations:
        raise KeyError("No activation recorded yet — run a stimulus first.")
    arr = np.asarray(lm.last_activations[node], dtype=np.float32)
    if arr.ndim >= 3:
        arr = arr[0, -1]                     # last-token logits for LMs
    arr = arr.reshape(-1)
    e = np.exp(arr - arr.max())
    p = e / e.sum()
    idx = np.argsort(-p)[:k]

    def label(i):
        if lm.labels and i < len(lm.labels):
            return lm.labels[i]
        if lm.tokenizer:
            return lm.tokenizer.id_to_token(int(i))
        return f"#{i}"
    return {"topk": [{"index": int(i), "label": label(i),
                      "prob": float(p[i]), "logit": float(arr[i])}
                     for i in idx]}

# ---------------------------------------------------------------------------
# Per-layer statistics detail (histogram + channel means)
# ---------------------------------------------------------------------------

def node_detail(lm: LoadedModel, node: str, bins: int = 28) -> dict:
    if node not in lm.last_activations:
        raise KeyError(f"No cached activation for '{node}' — run a stimulus.")
    a = _sanitize_array(np.asarray(lm.last_activations[node], dtype=np.float32)).ravel()
    if a.size > 200_000:                       # subsample huge tensors
        a = a[:: max(1, a.size // 200_000)]
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        hi = lo + 1e-6
    hist, edges = np.histogram(a, bins=bins, range=(lo, hi))
    full = np.asarray(lm.last_activations[node], dtype=np.float32)
    ch = None
    if full.ndim >= 2:                          # per-channel mean |x|
        ax = 1 if full.ndim >= 4 or full.shape[1] <= 4096 else full.ndim - 1
        move = np.moveaxis(full, ax, 0).reshape(full.shape[ax], -1)
        ch = np.abs(move).mean(axis=1)[:256].tolist()
    return {"hist": hist.tolist(),
            "edges": [_finite(edges[0]), _finite(edges[-1])],
            "channel_means": _finite(ch) if ch is not None else None,
            "n_active": int((np.abs(a) > 1e-6).sum()),
            "n_sampled": int(a.size)}

# ---------------------------------------------------------------------------
# Temporal fMRI — record activation timelines over a stimulus sequence
# ---------------------------------------------------------------------------

MAX_REC_FRAMES = 160


def _decode_video_frames(data: bytes) -> np.ndarray:
    import tempfile as _tf
    import imageio.v3 as iio
    with _tf.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(data)
        tmp = f.name
    return np.stack(list(iio.imiter(tmp, plugin="FFMPEG")))     # (F,H,W,3)


def _image_tensor_from_frame(frame: np.ndarray, shape: list[int],
                             normalize: str) -> np.ndarray:
    from PIL import Image
    if shape[1] in (1, 3):
        c, h, w, chw = shape[1], shape[2], shape[3], True
    else:
        h, w, c, chw = shape[1], shape[2], shape[3], False
    arr = np.asarray(Image.fromarray(frame).resize((w, h)),
                     dtype=np.float32) / 255.0
    if c == 1:
        arr = arr.mean(axis=2, keepdims=True)
    arr = apply_norm(arr, normalize, c)
    if chw:
        arr = arr.transpose(2, 0, 1)
    return arr[None, ...].astype(np.float32)


def is_diffusion(lm: LoadedModel) -> bool:
    """A denoising UNet, in latent OR pixel space.

    The general signature is: a timestep knob, something image-shaped to clean
    up, and an output shaped like that input (it predicts the noise). Requiring
    a *latent* would miss pixel-space DDPMs like ddpm-cifar10, which are real
    diffusion models that simply skip the VAE.
    """
    mods = [i.get("modality") for i in lm.inputs]
    if "scalar" not in mods:
        return False
    prim = next((i for i in lm.inputs if i["name"] == lm.primary_input), None)
    if prim is None or prim.get("modality") not in ("latent", "image"):
        return False
    outs = lm.graph.get("outputs") or []
    return any(len(o.get("shape") or []) == len(prim["shape"]) for o in outs)


def _scalar_input(lm: LoadedModel) -> dict | None:
    return next((i for i in lm.inputs if i.get("modality") == "scalar"), None)


#: The beta schedule is NOT in the ONNX file — it lives in the pipeline's
#: scheduler config, which we never see. Guessing wrong produces a plausible
#: but under-denoised trajectory, so it is exposed as a choice instead.
#: "scaled_linear" is Stable Diffusion; "linear" is the original DDPM/DDIM.
BETA_SCHEDULES = ("scaled_linear", "linear", "cosine")


def _alphas_cumprod(train_steps: int = 1000,
                    schedule: str = "scaled_linear") -> np.ndarray:
    if schedule == "linear":                      # Ho et al. DDPM
        betas = np.linspace(1e-4, 0.02, train_steps, dtype=np.float64)
    elif schedule == "cosine":                    # Nichol & Dhariwal
        s, t = 0.008, np.linspace(0, 1, train_steps + 1, dtype=np.float64)
        f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        ac = f / f[0]
        return np.clip(ac[1:], 1e-8, 1.0)
    else:                                         # Stable Diffusion
        betas = np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, train_steps,
                            dtype=np.float64) ** 2
    return np.cumprod(1.0 - betas)


def _ddim_trajectory(lm: LoadedModel, prim: dict, steps: int,
                     seed: int | None = None, train_steps: int = 1000,
                     schedule: str = "scaled_linear"):
    """Run a real DDIM (eta=0) denoising loop and return every latent along
    the way, plus the timestep used at each step.

    This is what makes the temporal carpet meaningful for a diffusion model:
    the sequence axis becomes NOISE LEVEL, so you can watch which layers carry
    the work early (coarse structure) versus late (fine detail).
    """
    ac = _alphas_cumprod(train_steps, schedule)
    ts = np.linspace(train_steps - 1, 0, steps).round().astype(int)
    rng = np.random.default_rng(7 if seed is None else int(seed))
    x = rng.standard_normal(prim["shape"]).astype(np.float32)
    tname = _scalar_input(lm)["name"]
    # the noise prediction is the model's declared output; lm.output_names is
    # the EXPOSED graph (every intermediate), so its last entry is not it.
    outs = lm.graph.get("outputs") or []
    out_name = outs[0]["name"] if outs else lm.output_names[-1]
    for o in outs:                       # prefer one shaped like the latent
        if len(o.get("shape") or []) == len(prim["shape"]):
            out_name = o["name"]
            break

    lats, used = [], []
    for i, t in enumerate(ts):
        lats.append(x.copy())
        used.append(int(t))
        feed = {prim["name"]: x.astype(np.float32),
                tname: np.array(float(t), np.float32)}
        _fill_companions(lm, feed)
        # the REAL graph output, not the last exposed intermediate
        eps = np.asarray(lm.session.run([out_name], feed)[0], np.float32)
        if eps.shape != x.shape:                 # not a noise-predictor
            break
        a_t = float(ac[t])
        a_prev = float(ac[ts[i + 1]]) if i + 1 < len(ts) else 1.0
        pred_x0 = (x - math.sqrt(1 - a_t) * eps) / math.sqrt(a_t)
        # sqrt(alpha_t) is ~0.07 at t=999, so a bad noise prediction gets
        # amplified ~15x per step and an untrained UNet diverges to inf within
        # a few steps. Real latents live in ~[-4, 4]; clamping wide enough to
        # never bind on a trained model keeps a broken one merely wrong
        # instead of numerically dead. (diffusers calls this clip_sample.)
        pred_x0 = np.clip(np.nan_to_num(pred_x0, nan=0.0, posinf=0.0,
                                        neginf=0.0), -10.0, 10.0)
        x = (math.sqrt(a_prev) * pred_x0
             + math.sqrt(max(1 - a_prev, 0.0)) * eps)     # DDIM, eta = 0
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return lats, used


def _prepare_recording(lm: LoadedModel, rec: dict) -> None:
    prim = next(i for i in lm.inputs if i["name"] == lm.primary_input)
    rec["prim"] = prim
    mode, T = rec["mode"], rec["frames"]
    if mode == "text":
        tok = lm.tokenizer or ByteTokenizer()
        lm.tokenizer = tok
        ids = tok.encode(rec.get("text") or "")
        if not ids:
            raise ValueError("Text tokenized to nothing.")
        rec["frames"] = min(T, len(ids))
        step = len(ids) / rec["frames"]
        rec["prefix_ends"] = [int((k + 1) * step) for k in range(rec["frames"])]
        rec["ids"] = ids
        rec["labels"] = ["…" + "".join(
            str(tok.id_to_token(i)) for i in
            ids[max(0, e - 4):e]) for e in rec["prefix_ends"]]
    elif mode == "video":
        if not rec.get("media"):
            raise ValueError("Video recording needs a video file.")
        frames = _decode_video_frames(rec["media"])
        shape = prim["shape"]
        if len(shape) == 5:                      # video-input model: window
            dims = shape[1:]
            c_axis = next((i for i, d in enumerate(dims) if d in (1, 3)), 0)
            Tm = dims[1] if c_axis == 0 else dims[0]
            rec["frames"] = min(T, max(2, len(frames) - Tm + 1))
            rec["win"] = Tm
        else:
            rec["frames"] = min(T, len(frames))
        idx = np.linspace(0, len(frames) - 1 - rec.get("win", 1) + 1,
                          rec["frames"]).astype(int)
        rec["starts"] = idx.tolist()
        rec["vframes"] = frames
        rec["labels"] = [f"frame {int(i)}" for i in idx]
    elif mode == "audio":
        if not rec.get("media"):
            raise ValueError("Audio recording needs an audio file.")
        sr = rec["sample_rate"]
        y = decode_audio(rec["media"], sr)
        shape, raw = prim["shape"], prim["raw_shape"]
        li = len(shape) - 1
        fixed = raw[li] if isinstance(raw[li], int) and raw[li] > 0 else 0
        win = fixed or max(sr // 2, 1600)
        if len(y) <= win:
            y = np.pad(y, (0, win + 1 - len(y)))
        rec["frames"] = min(T, 120)
        starts = np.linspace(0, len(y) - win, rec["frames"]).astype(int)
        rec["y"], rec["win"], rec["starts"] = y, win, starts.tolist()
        rec["labels"] = [f"{s/sr:.2f}s" for s in starts]
    elif mode == "denoise":
        if not is_diffusion(lm):
            raise ValueError("This model has no latent + timestep pair, so "
                             "there is no denoising loop to record.")
        rec["frames"] = min(T, 60)
        lats, used = _ddim_trajectory(lm, prim, rec["frames"], rec.get("seed"),
                                      schedule=rec.get("schedule",
                                                       "scaled_linear"))
        if len(lats) < 2:
            raise ValueError("The model did not return a noise prediction "
                             "shaped like its latent, so DDIM cannot step it.")
        rec["frames"] = len(lats)
        rec["latents"], rec["timesteps"] = lats, used
        rec["labels"] = [f"t={t}" for t in used]
    elif mode == "noise":
        rng = np.random.default_rng(rec.get("seed") or 7)
        dt = np.dtype(prim["dtype"])
        n_anchor = max(2, rec["frames"] // 24 + 1)
        if dt.kind == "f":
            rec["anchors"] = [rng.standard_normal(prim["shape"])
                              .astype(np.float32) for _ in range(n_anchor)]
        else:
            rec["anchors"] = [rng.integers(0, 100, size=prim["shape"])
                              .astype(dt) for _ in range(n_anchor)]
        rec["labels"] = [f"step {k}" for k in range(rec["frames"])]
    else:
        raise ValueError(f"Unknown recording mode: {mode}")


def _frame_feed(lm: LoadedModel, rec: dict, k: int) -> dict:
    prim, mode = rec["prim"], rec["mode"]
    if mode == "denoise":
        k = max(0, min(k, len(rec["latents"]) - 1))
        feed = {prim["name"]: rec["latents"][k].astype(np.float32),
                _scalar_input(lm)["name"]:
                    np.array(float(rec["timesteps"][k]), np.float32)}
        _fill_companions(lm, feed)
        return feed
    if mode == "text":
        return make_multi_feed(lm, {prim["name"]: {
            "mode": "tokens", "values": rec["ids"][:rec["prefix_ends"][k]]}})
    if mode == "video":
        shape = prim["shape"]
        s = rec["starts"][k]
        if len(shape) == 5:
            sub = rec["vframes"][s:s + rec["win"]]
            dims = shape[1:]
            c_axis = next((i for i, d in enumerate(dims) if d in (1, 3)), 0)
            layout = "NCTHW" if c_axis == 0 else \
                     ("NTHWC" if dims[-1] in (1, 3) else "NTCHW")
            C, Tm, H, W = (dims if layout == "NCTHW" else
                           (dims[1], dims[0], dims[2], dims[3])
                           if layout == "NTCHW"
                           else (dims[3], dims[0], dims[1], dims[2]))
            from PIL import Image
            sel = np.stack([np.asarray(Image.fromarray(f).resize((W, H)))
                            for f in sub]).astype(np.float32) / 255.0
            if len(sel) < Tm:
                sel = np.concatenate([sel, np.repeat(sel[-1:],
                                     Tm - len(sel), 0)])
            if C == 1:
                sel = sel.mean(axis=3, keepdims=True)
            x = (sel.transpose(3, 0, 1, 2) if layout == "NCTHW" else
                 sel.transpose(0, 3, 1, 2) if layout == "NTCHW" else sel)
            x = x[None, ...].astype(np.float32)
        else:
            x = _image_tensor_from_frame(rec["vframes"][s], shape,
                                         rec["normalize"])
        feed = {prim["name"]: x}
        _fill_companions(lm, feed)
        return feed
    if mode == "audio":
        s = rec["starts"][k]
        x = fit_audio(rec["y"][s:s + rec["win"]], prim["shape"],
                      prim["raw_shape"], rec["sample_rate"])
        feed = {prim["name"]: x}
        _fill_companions(lm, feed)
        return feed
    if mode == "noise":
        A = rec["anchors"]
        seg = (len(A) - 1) * k / max(rec["frames"] - 1, 1)
        i = min(int(seg), len(A) - 2)
        t = seg - i
        if A[0].dtype.kind == "f":
            th = t * math.pi / 2                    # slerp keeps norm ~constant
            x = A[i] * math.cos(th) + A[i + 1] * math.sin(th)
        else:                                       # int inputs: crossfade mask
            rng = np.random.default_rng(1000 + k)
            m = rng.random(A[i].shape) < t
            x = np.where(m, A[i + 1], A[i])
        feed = {prim["name"]: x.astype(A[0].dtype)}
        _fill_companions(lm, feed)
        return feed
    raise ValueError(mode)


def start_recording(lm: LoadedModel, mode: str, frames: int, media=None,
                    text=None, sample_rate=16000, normalize="unit",
                    seed=None, schedule="scaled_linear") -> dict:
    rec = {"mode": mode, "frames": int(min(max(frames, 2), MAX_REC_FRAMES)),
           "media": media, "text": text, "sample_rate": int(sample_rate),
           "normalize": normalize, "seed": seed, "labels": [],
           "schedule": schedule}
    _prepare_recording(lm, rec)
    timeline: dict[str, list] = {}
    global_e = []
    attn_ids = [n["id"] for n in lm.graph["nodes"] if n.get("attention")]
    attn_frames: dict[str, list] = {a: [] for a in attn_ids}
    node_ids = [n["id"] for n in lm.graph["nodes"]]
    for k in range(rec["frames"]):
        feed = _frame_feed(lm, rec, k)
        outs = lm.session.run(lm.output_names, feed)
        acts = dict(zip(lm.output_names, outs))
        ge, n = 0.0, 0
        for nid in node_ids:
            a = acts.get(nid)
            if not isinstance(a, np.ndarray) or a.dtype.kind not in "fiu":
                continue
            if a.size == 0:
                continue
            e = _finite(np.abs(a.astype(np.float32)).mean())
            timeline.setdefault(nid, []).append(round(e, 5))
            ge += e
            n += 1
        for aid in attn_ids:                     # head-averaged matrix per frame
            m = _as_attention(acts.get(aid))
            if m is not None:
                am = m.mean(axis=0)
                if am.shape[0] > 24 or am.shape[1] > 24:
                    qi = np.linspace(0, am.shape[0]-1, min(24, am.shape[0])).astype(int)
                    ki = np.linspace(0, am.shape[1]-1, min(24, am.shape[1])).astype(int)
                    am = am[np.ix_(qi, ki)]
                attn_frames[aid].append(np.round(am, 4).tolist())
            else:
                attn_frames[aid].append(None)
        global_e.append(round(ge / max(n, 1), 5))
    allv = np.concatenate([np.asarray(v) for v in timeline.values()]) \
        if timeline else np.zeros(1)
    lm.recording = rec
    return {"frames": rec["frames"], "timeline": timeline,
            "global": global_e, "labels": rec["labels"],
            "attention": {"nodes": attn_ids, "frames": attn_frames},
            "scale": float(np.percentile(allv, 99)) or 1.0}


def seek_frame(lm: LoadedModel, k: int) -> dict:
    rec = getattr(lm, "recording", None)
    if not rec:
        raise KeyError("No recording — record a sequence first.")
    k = max(0, min(int(k), rec["frames"] - 1))
    return run_stimulus(lm, _frame_feed(lm, rec, k))

# ---------------------------------------------------------------------------
# Attention flow — extract [head, query, key] weight matrices
# ---------------------------------------------------------------------------

def _as_attention(arr: np.ndarray) -> np.ndarray | None:
    """Normalize an attention activation to (heads, Q, K). Accepts
    (B,H,Q,K), (H,Q,K), (B,Q,K) or (Q,K)."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 4:
        a = a[0]                       # (H, Q, K)
    elif a.ndim == 3:
        # (B, Q, K) single head, or already (H, Q, K)
        if a.shape[1] == a.shape[2]:   # square last two → treat dim0 as heads
            pass
        else:
            a = a[0][None, ...]
    elif a.ndim == 2:
        a = a[None, ...]
    else:
        return None
    if a.ndim != 3 or a.shape[-1] < 1:
        return None
    return a


def attention_map(lm: LoadedModel, node: str, head: int | None = None,
                  max_tok: int = 64) -> dict:
    if node not in lm.last_activations:
        raise KeyError("No activation for this attention layer — run a "
                       "stimulus first.")
    nrec = next((n for n in lm.graph["nodes"] if n["id"] == node), {})
    a = _as_attention(_sanitize_array(np.asarray(lm.last_activations[node], dtype=np.float32)))
    if a is None or nrec.get("attention_fused"):
        # Fused / Flash-Attention op: the [Q,K] matrix was never materialized.
        # Fall back to the fused output's per-token magnitude — which tokens
        # the attention block emitted strong signal for.
        out = np.asarray(lm.last_activations[node], dtype=np.float32)
        if out.ndim >= 3:                # (B, T, D) → per-token L2
            tok_energy = np.linalg.norm(out[0], axis=-1)
        elif out.ndim == 2:
            tok_energy = np.linalg.norm(out, axis=-1)
        else:
            tok_energy = np.abs(out.ravel())
        toks = lm.last_tokens or []
        return {"fused": True, "heads": nrec.get("num_heads", 0),
                "q": int(tok_energy.shape[0]), "k": int(tok_energy.shape[0]),
                "token_energy": np.round(tok_energy[:max_tok], 5).tolist(),
                "tokens": [str(t["token"]) for t in toks][:max_tok],
                "note": ("Fused attention op — the query×key matrix is never "
                         "materialized, so per-token output energy is shown "
                         "instead of the attention pattern.")}
    H, Q, K = a.shape
    mat = a.mean(axis=0) if head is None else a[max(0, min(head, H - 1))]
    # downsample if huge
    def _ds(m, n):
        if m.shape[0] <= n and m.shape[1] <= n:
            return m, list(range(m.shape[0])), list(range(m.shape[1]))
        qi = np.linspace(0, m.shape[0] - 1, min(n, m.shape[0])).astype(int)
        ki = np.linspace(0, m.shape[1] - 1, min(n, m.shape[1])).astype(int)
        return m[np.ix_(qi, ki)], qi.tolist(), ki.tolist()
    mat_ds, qi, ki = _ds(mat, max_tok)
    toks = lm.last_tokens or []
    return {"heads": H, "q": Q, "k": K, "head": head,
            "matrix": np.round(mat_ds, 5).tolist(),
            "q_idx": qi, "k_idx": ki,
            "tokens": [str(t["token"]) for t in toks][:max(Q, K)]}


def attention_nodes(lm: LoadedModel) -> list[str]:
    return [n["id"] for n in lm.graph["nodes"] if n.get("attention")]

# ---------------------------------------------------------------------------
# Stimulus viewer — render the actual input that produced the activations
# ---------------------------------------------------------------------------

_NORM_INV = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "clip":     ([0.48145466, 0.4578275, 0.40821073],
                 [0.26862954, 0.26130258, 0.27577711]),
    "openai":   ([0.48145466, 0.4578275, 0.40821073],
                 [0.26862954, 0.26130258, 0.27577711]),
    "inception":([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
}


def stimulus_info(lm: LoadedModel) -> dict:
    """What kind of stimulus is available to view, per input."""
    prev = lm.last_stimulus or {}
    out = []
    for inp in stimulus_inputs(lm):
        p = prev.get(inp["name"], {})
        out.append({"name": inp["name"], "modality": inp.get("modality"),
                    "mode": p.get("mode"), "shape": inp["shape"]})
    return {"inputs": out, "tokens": lm.last_tokens}


def _denorm_image(arr: np.ndarray, normalize: str) -> np.ndarray:
    if normalize == "amax":     # synthesized input: rescale for viewing
        lo, hi = float(arr.min()), float(arr.max())
        return (arr - lo) / (hi - lo + 1e-8)
    inv = _NORM_INV.get(normalize)
    if normalize == "signed":
        return (arr + 1) / 2
    if inv and arr.shape[-1] == 3:
        return arr * np.array(inv[1], np.float32) + np.array(inv[0], np.float32)
    return arr


def stimulus_image(lm: LoadedModel, input_name: str | None = None,
                   frame: int | None = None) -> bytes:
    """Render the stored input tensor for `input_name` back to a PNG."""
    from PIL import Image
    feed = lm.last_feed or {}
    name = input_name or lm.primary_input
    if name not in feed:
        raise KeyError("No stimulus recorded for this input — run one first.")
    x = np.asarray(feed[name], dtype=np.float32)
    normalize = (lm.last_stimulus or {}).get(name, {}).get("_normalize", "unit")
    # locate a viewable image inside the tensor
    if x.ndim == 5:                         # video (N,C,T,H,W) or (N,T,H,W,C)
        dims = x.shape[1:]
        c_axis = next((i for i, d in enumerate(dims) if d in (1, 3)), 0)
        if c_axis == 0:                     # NCTHW
            t = x[0].transpose(1, 2, 3, 0)  # T,H,W,C
        elif dims[-1] in (1, 3):            # NTHWC
            t = x[0]
        else:                               # NTCHW
            t = x[0].transpose(0, 2, 3, 1)
        fr = 0 if frame is None else max(0, min(frame, t.shape[0] - 1))
        img = t[fr]
    elif x.ndim == 4:                       # image (N,C,H,W) or (N,H,W,C)
        img = x[0].transpose(1, 2, 0) if x.shape[1] in (1, 3) else x[0]
    elif x.ndim == 3:                       # (N,mel,frames) audio spectrogram
        img = np.abs(x[0])[..., None].repeat(3, -1)
        img = (img - img.min()) / (img.ptp() + 1e-8)
        return _to_png(img)
    else:
        raise ValueError("This input has no viewable image form.")
    if img.shape[-1] == 1:
        img = img.repeat(3, -1)
    img = _denorm_image(img, normalize)
    img = np.clip(img, 0, 1)
    return _to_png(img)


def _to_png(img01: np.ndarray) -> bytes:
    from PIL import Image
    pil = Image.fromarray((img01 * 255).astype(np.uint8))
    if max(pil.size) < 224:
        f = max(1, 224 // max(pil.size))
        pil = pil.resize((pil.width * f, pil.height * f), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def stimulus_waveform(lm: LoadedModel, input_name: str | None = None,
                      points: int = 800) -> dict:
    """Return a downsampled waveform (or per-frame mel energy) for plotting."""
    feed = lm.last_feed or {}
    name = input_name or lm.primary_input
    if name not in feed:
        raise KeyError("No stimulus recorded — run one first.")
    x = np.asarray(feed[name], dtype=np.float32)
    if x.ndim <= 2 or (x.ndim == 3 and x.shape[1] == 1):     # raw waveform
        y = x.reshape(-1)
        kind = "waveform"
    elif x.ndim in (3, 4):                                   # mel: energy/frame
        m = np.abs(x[0]) if x.ndim == 3 else np.abs(x[0, 0])
        y = m.mean(axis=0) if m.shape[0] < m.shape[1] else m.mean(axis=1)
        kind = "mel-energy"
    else:
        raise ValueError("No waveform in this input.")
    n = y.shape[0]
    if n > points:
        idx = np.linspace(0, n - 1, points).astype(int)
        y = y[idx]
    return {"kind": kind, "values": [_finite(v) for v in y],
            "n": int(n)}

# ---------------------------------------------------------------------------
# Jacobian lens (J-lens) — read what a layer is "disposed to say"
#
# Adapts Anthropic's jacobian-lens (github.com/anthropics/jacobian-lens) to
# ONNX. Their transport is lens_l(h) = unembed(J_l @ h), J_l = E[dh_final/dh_l].
# We can't autodiff an arbitrary ONNX graph, so J_l is estimated by finite
# differences: run the model forward from layer l to the logits with the
# activation perturbed along random directions, and least-squares fit the
# linear map J_l : h_l -> logits. Averaging over sample stimuli gives the
# "averaged" lens; a single stimulus gives the "fast" lens.
# ---------------------------------------------------------------------------

JLENS_MAX_HIDDEN = 1024     # cap on layer width for the finite-diff probe
# The finite-difference lens (and circuit tracing) rebuild a forward subgraph in
# memory: proto + subgraph copy + serialize + session runs to roughly 3-4x the
# model's on-disk size. Past this we refuse with an explanation instead of
# letting the OS OOM-kill the server.
SUBGRAPH_MAX_MB = 150       # fallback cap when free RAM can't be measured
# Measured, not guessed: rebuilding the subgraph and fitting the Jacobian costs
# ~4.9x the model size on BERT-tiny and ~5.2x on ViT-tiny (proto cache +
# subgraph copy + serialize + session + the d x vocab Jacobian). 5.5 leaves a
# margin. Being too permissive here OOM-kills the server, which is far worse
# than refusing — so the estimate errs high on purpose.
SUBGRAPH_RAM_FACTOR = 5.5


def _avail_mb() -> float:
    """Free RAM in MB, or 0 if we can't tell (then the fixed cap applies)."""
    try:                                   # Linux / WSL
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    try:                                   # anything with psutil
        import psutil
        return psutil.virtual_memory().available / 1e6
    except Exception:
        return 0.0
JLENS_MAX_VOCAB = 4096      # cap on logit dim we fit against


def _logits_output(lm: LoadedModel) -> str:
    """The graph output that carries logits/class scores."""
    outs = lm.graph["outputs"]
    best, best_dim = None, -1
    for o in outs:
        sh = [d for d in o["shape"] if isinstance(d, int) and d > 0]
        if sh and sh[-1] > best_dim:
            best, best_dim = o["name"], sh[-1]
    return best or (outs[0]["name"] if outs else lm.output_names[-1])


def _lens_decodable(lm: LoadedModel) -> tuple[bool, str, int]:
    """Can this model's output be decoded to labels/tokens?

    Returns (ok, reason, out_dim). Decodable when the output last-dim matches
    the tokenizer vocab or the attached class labels — otherwise the output is
    a hidden/feature vector and the readout would be meaningless."""
    name = _logits_output(lm)
    out = next((o for o in lm.graph["outputs"] if o["name"] == name), None)
    dim = 0
    if out:
        sh = [d for d in out["shape"] if isinstance(d, int) and d > 0]
        dim = sh[-1] if sh else 0
    # class-label head
    if lm.labels and dim and abs(dim - len(lm.labels)) <= 1:
        return True, "labels", dim
    # vocab head: last dim matches a tokenizer vocab (or is large) with a tokenizer
    if lm.tokenizer is not None and dim >= 64:
        return True, "vocab", dim
    # image/audio class head with modest class count and labels
    if lm.labels and dim:
        return True, "labels", dim
    if dim >= 1000 and lm.tokenizer is None:
        return True, "vocab_nolabels", dim
    return (False,
            ("This model's final output is a feature/hidden vector "
             f"(dim {dim}), not a vocabulary or labelled class head — the "
             "lens has nothing to decode into. Load a model with a "
             "classification or language-model head (e.g. ViT, DistilBERT-SST2 "
             "with labels, a GPT), or attach class labels."),
            dim)


def _build_subgraph_session(lm: LoadedModel, node: str, logits_name: str):
    """A session that takes node's activation as input and outputs logits.

    Any tensor a kept node needs that isn't produced inside the subgraph (and
    isn't an initializer) is exposed as an additional graph input, so we can
    feed the recorded activations for those too. This keeps the cut clean even
    when a downstream node also depends on something before the cut point."""
    import onnxruntime as ort
    try:
        mb = lm.onnx_path.stat().st_size / 1e6
    except Exception:
        mb = 0
    # Rebuilding a forward subgraph costs roughly SUBGRAPH_RAM_FACTOR x the
    # model size (proto + subgraph copy + serialize + session). Rather than a
    # fixed cap, check what this machine actually has free — a big model is
    # fine on a workstation and fatal in a small container.
    need = mb * SUBGRAPH_RAM_FACTOR
    avail = _avail_mb()
    if (avail and need > avail * 0.7) or (not avail and mb > SUBGRAPH_MAX_MB):
        where = (f"only {avail:.0f} MB is free" if avail
                 else "free memory couldn't be measured")
        raise ValueError(
            f"This model is {mb:.0f} MB. The lens and circuit tracer rebuild a "
            f"forward subgraph in memory (~{need:.0f} MB peak) and {where}, so "
            f"this would exhaust RAM. Free some memory or use a smaller model "
            f"(the demo samples, BERT-tiny and ViT-tiny all work). Everything "
            f"else — activations, attention, temporal, attribution, "
            f"maximization, health, latency, export — works at any size.")
    if getattr(lm, "_proto_cache", None) is None:
        p = onnx.load(str(lm.onnx_path))
        lm._proto_cache = shape_inference.infer_shapes(p)
    proto = lm._proto_cache
    g = proto.graph

    prod = {}
    for i, nd in enumerate(g.node):
        for o in nd.output:
            prod[o] = i
    if node not in prod:
        raise ValueError(f"'{node}' is not a produced tensor.")

    # walk back from logits, stopping at `node` (our cut tensor)
    needed, stack = set(), [logits_name]
    while stack:
        t = stack.pop()
        if t == node:
            continue
        pi = prod.get(t)
        if pi is None or pi in needed:
            continue
        needed.add(pi)
        for inp in g.node[pi].input:
            stack.append(inp)
    keep = sorted(needed)

    init_names = {ini.name for ini in g.initializer}
    produced_in_sub = {o for i in keep for o in g.node[i].output}
    vi_map = {vi.name: vi for vi in list(g.value_info) + list(g.output)
              + list(g.input)}

    def _vi(name):
        return vi_map.get(name) or onnx.helper.make_tensor_value_info(
            name, onnx.TensorProto.FLOAT, None)

    # external inputs = tensors kept nodes need but that aren't produced here
    # or initializers. Always includes `node`; may include original graph
    # inputs (which we can feed from the recorded feed).
    ext_inputs = []
    seen = set()
    for i in keep:
        for inp in g.node[i].input:
            if (inp and inp not in produced_in_sub and inp not in init_names
                    and inp not in seen):
                seen.add(inp)
                ext_inputs.append(inp)
    if node not in seen:
        ext_inputs.insert(0, node)

    inits = [ini for ini in g.initializer
             if ini.name in {inp for i in keep for inp in g.node[i].input}]

    sub = onnx.helper.make_graph(
        [g.node[i] for i in keep], f"sub_{node[:8]}",
        [_vi(n) for n in ext_inputs], [_vi(logits_name)], inits)
    m = onnx.helper.make_model(sub, opset_imports=list(proto.opset_import))
    m.ir_version = proto.ir_version
    try:
        sess = ort.InferenceSession(m.SerializeToString(),
                                    providers=["CPUExecutionProvider"])
        return sess, node, logits_name, ext_inputs
    except Exception as e:
        raise ValueError(f"Could not isolate a forward path from '{node}' to "
                         f"logits: {str(e)[:160]}")


def _ext_feed(lm: LoadedModel, ext_inputs: list[str], cut_node: str) -> dict:
    """Values for the subgraph's non-cut external inputs, from what we ran."""
    feed = {}
    for name in ext_inputs:
        if name == cut_node:
            continue
        if name in (lm.last_activations or {}):
            feed[name] = np.asarray(lm.last_activations[name])
        elif name in (lm.last_feed or {}):
            feed[name] = np.asarray(lm.last_feed[name])
    return feed


def _fit_jacobian(sess, in_name, out_name, h0: np.ndarray, ext: dict,
                  full_act: np.ndarray, pos_index, n_probe: int = 48,
                  eps: float = 0.02):
    """Least-squares J so logits ≈ J @ h around h0, injecting h at position.

    full_act is the recorded activation tensor for the cut node; we replace the
    slice at pos_index with the (perturbed) h before running the subgraph, so
    the other positions stay at their true values."""
    d = h0.shape[0]

    def run(hvec):
        act = full_act.copy()
        _set_pos(act, pos_index, hvec)
        fd = dict(ext); fd[in_name] = act.astype(np.float32)
        return sess.run([out_name], fd)[0].reshape(-1)

    base = run(h0)
    v = base.shape[0]
    n_probe = min(n_probe, d)
    rng = np.random.default_rng(0)
    D = rng.standard_normal((n_probe, d)).astype(np.float32)
    D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
    dY = np.zeros((n_probe, v), np.float32)
    scale = eps * (np.linalg.norm(h0) / np.sqrt(d) + 1e-6)
    for i in range(n_probe):
        dY[i] = (run(h0 + scale * D[i]) - base) / scale
    Jt, *_ = np.linalg.lstsq(D, dY, rcond=None)
    return Jt, base


def _set_pos(act: np.ndarray, pos_index, hvec: np.ndarray) -> None:
    kind, pos = pos_index
    if kind == "token" and act.ndim == 3:
        act[0, pos] = hvec
    elif kind == "chanpool" and act.ndim == 4:      # broadcast over H,W
        act[0] = hvec[:, None, None]
    elif kind == "vec" and act.ndim == 2:
        act[0] = hvec
    else:
        act.reshape(-1)[:hvec.shape[0]] = hvec


def _hidden_at(lm: LoadedModel, node: str, position: int | None):
    """Extract the hidden vector at `node` for a given sequence position.
    Returns (h_vector, meta, full_activation, pos_index)."""
    a = np.asarray(lm.last_activations[node], dtype=np.float32)
    if a.ndim == 3:            # (B, T, D)
        pos = a.shape[1] - 1 if position is None else max(0, min(position, a.shape[1]-1))
        return a[0, pos], ("token", pos, a.shape[1]), a, ("token", pos)
    if a.ndim == 4:            # (B, C, H, W) → global-avg-pooled channel vector
        return a[0].mean(axis=(1, 2)), ("chanpool", 0, 1), a, ("chanpool", 0)
    if a.ndim == 2:            # (B, D)
        return a[0], ("vec", 0, 1), a, ("vec", 0)
    return a.reshape(-1), ("flat", 0, 1), a, ("flat", 0)


def jacobian_lens(lm: LoadedModel, node: str, position: int | None = None,
                  k: int = 12, n_probe: int = 48) -> dict:
    """Read out top-k vocab/classes this layer is disposed toward."""
    if node not in lm.last_activations:
        raise KeyError("Run a stimulus first, then apply the lens.")
    ok, reason, dim = _lens_decodable(lm)
    if not ok:
        raise ValueError(reason)
    logits_name = _logits_output(lm)
    h, meta, full_act, pos_index = _hidden_at(lm, node, position)
    if h.shape[0] > JLENS_MAX_HIDDEN:
        raise ValueError(f"Layer width {h.shape[0]} exceeds the lens probe cap "
                         f"({JLENS_MAX_HIDDEN}). Pick a narrower layer.")
    sess, in_name, out_name, ext_inputs = _build_subgraph_session(
        lm, node, logits_name)
    ext = _ext_feed(lm, ext_inputs, node)
    Jt, base = _fit_jacobian(sess, in_name, out_name, h, ext, full_act,
                             pos_index, n_probe)
    lens_logits = h @ Jt                              # transported logits
    e = np.exp(lens_logits - lens_logits.max())
    p = e / (e.sum() + 1e-9)
    idx = np.argsort(-p)[:k]

    def label(i):
        if lm.labels and i < len(lm.labels):
            return lm.labels[i]
        if lm.tokenizer:
            return lm.tokenizer.id_to_token(int(i))
        return f"#{i}"
    return {"node": node, "position": meta[1], "pos_kind": meta[0],
            "seq_len": meta[2], "logits_dim": int(lens_logits.shape[0]),
            "topk": [{"index": int(i), "label": label(i),
                      "prob": _finite(p[i]), "logit": _finite(lens_logits[i])}
                     for i in idx]}


def jacobian_lens_stack(lm: LoadedModel, k: int = 5, n_probe: int = 32,
                        position: int | None = None, max_layers: int = 14) -> dict:
    """Apply the lens at a sampled set of layers to build the layer × readout
    view (like Anthropic's slice page: what each depth is 'thinking')."""
    ok, reason, dim = _lens_decodable(lm)
    if not ok:
        raise ValueError(reason)
    logits_name = _logits_output(lm)
    # candidate layers: float activations, right rank, not too wide
    cand = []
    for n in sorted(lm.graph["nodes"], key=lambda x: x["depth"]):
        a = lm.last_activations.get(n["id"])
        if not isinstance(a, np.ndarray) or a.dtype.kind != "f":
            continue
        if a.ndim not in (2, 3, 4):
            continue
        w = a.shape[-1] if a.ndim in (2, 3) else a.shape[1]
        if 2 <= w <= JLENS_MAX_HIDDEN:
            cand.append(n)
    if len(cand) > max_layers:                    # even sample across depth
        idx = np.linspace(0, len(cand) - 1, max_layers).astype(int)
        cand = [cand[i] for i in idx]
    rows = []
    for n in cand:
        node = n["id"]
        try:
            h, meta, full_act, pos_index = _hidden_at(lm, node, position)
            if h.shape[0] > JLENS_MAX_HIDDEN or h.shape[0] < 2:
                continue
            sess, in_name, out_name, ext_inputs = _build_subgraph_session(
                lm, node, logits_name)
            ext = _ext_feed(lm, ext_inputs, node)
            Jt, base = _fit_jacobian(sess, in_name, out_name, h, ext,
                                     full_act, pos_index, n_probe)
            ll = h @ Jt
            top = np.argsort(-ll)[:k]

            def label(i):
                if lm.labels and i < len(lm.labels):
                    return lm.labels[i]
                if lm.tokenizer:
                    return lm.tokenizer.id_to_token(int(i))
                return f"#{i}"
            rows.append({"node": node, "op": n["op"], "depth": n["depth"],
                         "top": [label(int(i)) for i in top]})
        except Exception:
            continue
    return {"rows": rows, "position": position}

# ---------------------------------------------------------------------------
# Neuron attribution — what makes a unit fire?
#   (A) occlusion saliency: perturb regions of the input, measure the drop in
#       the target unit's activation → a saliency map over the input.
#   (B) stimulus ranking: run a set of stimuli, rank by how hard the unit fires.
# The target scalar is the mean |activation| of the chosen channel/unit (or the
# whole layer if no channel is given).
# ---------------------------------------------------------------------------

ATTR_MAX_CELLS = 196          # cap occlusion probes per axis-grid (14x14)


def _target_scalar(act: np.ndarray, channel: int | None) -> float:
    """Scalar response of the chosen unit from a layer activation."""
    a = np.asarray(act, dtype=np.float32)
    if channel is None:
        return float(np.abs(a).mean())
    if a.ndim == 4:                      # (B,C,H,W)
        c = min(channel, a.shape[1] - 1)
        return float(np.abs(a[0, c]).mean())
    if a.ndim == 3:                      # (B,T,D) → channel = feature dim
        c = min(channel, a.shape[2] - 1)
        return float(np.abs(a[0, :, c]).mean())
    if a.ndim == 2:                      # (B,D)
        c = min(channel, a.shape[1] - 1)
        return float(abs(a[0, c]))
    return float(np.abs(a).mean())


def _run_feed_get(lm: LoadedModel, feed: dict, node: str) -> np.ndarray:
    out = lm.session.run([node] if node in lm.output_names else lm.output_names,
                         feed)
    if node in lm.output_names and len(out) == 1:
        return out[0]
    acts = dict(zip(lm.output_names, out))
    return acts[node]


def attribution_occlusion(lm: LoadedModel, node: str, channel: int | None,
                          input_name: str | None = None, grid: int = 10,
                          frame: int | None = None) -> dict:
    """Occlude patches/tokens/windows of the input; report the resulting drop
    in the unit's activation as a saliency map aligned to the stimulus."""
    if lm.last_feed is None or node not in lm.last_activations:
        raise KeyError("Run a stimulus first, then attribute.")
    name = input_name or lm.primary_input
    if name not in lm.last_feed:
        raise KeyError("No recorded stimulus for this input.")
    base_feed = {k: np.array(v, copy=True) for k, v in lm.last_feed.items()}
    x = base_feed[name].astype(np.float32)
    base_resp = _target_scalar(lm.last_activations[node], channel)

    modality = next((i.get("modality") for i in lm.inputs if i["name"] == name),
                    "tensor")

    def resp_with(mod_x):
        f = dict(base_feed); f[name] = mod_x.astype(base_feed[name].dtype)
        return _target_scalar(_run_feed_get(lm, f, node), channel)

    result = {"node": node, "channel": channel, "input": name,
              "modality": modality, "base": _finite(base_resp)}

    # ---- image: occlude a grid of patches with the mean value ----
    if x.ndim == 4:
        chw = x.shape[1] in (1, 3)
        H, W = (x.shape[2], x.shape[3]) if chw else (x.shape[1], x.shape[2])
        g = min(grid, 14)
        fill = float(x.mean())
        sal = np.zeros((g, g), np.float32)
        hs, ws = max(1, H // g), max(1, W // g)
        for i in range(g):
            for j in range(g):
                xm = x.copy()
                y0, y1 = i * hs, min(H, (i + 1) * hs)
                x0, x1 = j * ws, min(W, (j + 1) * ws)
                if chw:
                    xm[0, :, y0:y1, x0:x1] = fill
                else:
                    xm[0, y0:y1, x0:x1, :] = fill
                sal[i, j] = base_resp - resp_with(xm)     # drop = importance
        result.update(kind="image-grid", grid=g,
                      saliency=np.round(sal, 5).tolist())
        return result

    # ---- text: occlude each token (replace with pad/zero) ----
    if x.ndim == 2 and modality == "text":
        T = x.shape[1]
        toks = lm.last_tokens or []
        n = min(T, len(toks) or T)
        pad_id = 0
        drops = []
        for t in range(n):
            xm = x.copy(); xm[0, t] = pad_id
            drops.append(_finite(base_resp - resp_with(xm)))
        result.update(kind="tokens",
                      tokens=[str(tk["token"]) for tk in toks][:n],
                      saliency=drops)
        return result

    # ---- audio / 1-D: occlude sliding windows ----
    if x.ndim <= 3:
        flat_axis = x.shape[-1]
        g = min(grid, 24)
        win = max(1, flat_axis // g)
        sal = []
        for i in range(g):
            xm = x.copy()
            s = i * win; e = min(flat_axis, s + win)
            xm[..., s:e] = 0.0
            sal.append(_finite(base_resp - resp_with(xm)))
        result.update(kind="windows", grid=g, saliency=sal)
        return result

    raise ValueError("This input shape isn't supported for occlusion.")

def attribution_rank_frames(lm: LoadedModel, node: str, channel: int | None,
                            top: int = 8) -> dict:
    """If a temporal recording exists, rank its frames by how hard the unit
    fires — the frames become an empirical 'preferred stimulus' ranking."""
    rec = getattr(lm, "recording", None)
    if not rec:
        raise KeyError("No recorded sequence — record a stimulus sequence "
                       "first (temporal panel), then rank its frames.")
    responses = []
    for k in range(rec["frames"]):
        feed = _frame_feed(lm, rec, k)
        act = _run_feed_get(lm, feed, node)
        responses.append(_target_scalar(act, channel))
    order = list(np.argsort(-np.asarray(responses)))
    labels = rec.get("labels") or [f"frame {k}" for k in range(rec["frames"])]
    ranked = [{"frame": int(k), "label": labels[k],
               "response": _finite(responses[k])} for k in order[:top]]
    return {"node": node, "channel": channel, "count": rec["frames"],
            "ranked": ranked,
            "series": [_finite(r) for r in responses],
            "labels": labels}


def attribution_rank_noise(lm: LoadedModel, node: str, channel: int | None,
                           n: int = 24, top: int = 6) -> dict:
    """Rank a batch of random noise stimuli by how hard the unit fires —
    a quick empirical probe of what excites the unit for any model."""
    prim = next(i for i in lm.inputs if i["name"] == lm.primary_input)
    dt = np.dtype(prim["dtype"])
    responses, seeds = [], []
    for s in range(n):
        rng = np.random.default_rng(1000 + s)
        if dt.kind == "f":
            xv = rng.standard_normal(prim["shape"]).astype(np.float32)
        else:
            xv = rng.integers(0, 100, size=prim["shape"]).astype(dt)
        feed = {prim["name"]: xv}
        _fill_companions(lm, feed)
        act = _run_feed_get(lm, feed, node)
        responses.append(_target_scalar(act, channel))
        seeds.append(1000 + s)
    order = list(np.argsort(-np.asarray(responses)))
    ranked = [{"seed": int(seeds[k]), "response": _finite(responses[k])}
              for k in order[:top]]
    return {"node": node, "channel": channel, "count": n, "ranked": ranked,
            "best_seed": int(seeds[order[0]]),
            "worst_seed": int(seeds[order[-1]])}

# ---------------------------------------------------------------------------
# Circuit tracing — causal influence between layers
#   Perturb a source unit's activation, run forward, and measure how much each
#   downstream unit changes. Reuses the subgraph runner (source = cut tensor,
#   target = any downstream tensor). Two modes:
#     (A) trace: source -> per-channel influence at a chosen target layer
#     (B) ablate: zero the source unit, measure impact on the final output
# ---------------------------------------------------------------------------

def _channel_vector(act: np.ndarray, channel: int | None) -> np.ndarray:
    """Per-channel summary vector of a layer activation (for target readout)."""
    a = np.asarray(act, dtype=np.float32)
    if a.ndim == 4:                      # (B,C,H,W) -> mean|.| per channel
        return np.abs(a[0]).mean(axis=(1, 2))
    if a.ndim == 3:                      # (B,T,D) -> mean|.| per feature dim
        return np.abs(a[0]).mean(axis=0)
    if a.ndim == 2:                      # (B,D)
        return np.abs(a[0])
    return np.abs(a).reshape(-1)


def _depth_of(lm: LoadedModel, node: str) -> int:
    for n in lm.graph["nodes"]:
        if n["id"] == node:
            return n["depth"]
    return -1


def circuit_trace(lm: LoadedModel, source: str, target: str,
                  source_channel: int | None = None, mode: str = "boost",
                  strength: float = 2.0, top: int = 12) -> dict:
    """Perturb the source unit and measure per-channel change at the target.

    mode: 'boost' scales the source channel up by `strength`; 'ablate' zeros it.
    Returns the target channels that move most (the causal downstream wire)."""
    if source not in lm.last_activations or target not in lm.last_activations:
        raise KeyError("Run a stimulus first, then trace.")
    if _depth_of(lm, target) <= _depth_of(lm, source):
        raise ValueError("Target must be deeper than the source layer — "
                         "influence flows forward.")
    # baseline target activation (recorded)
    base_target = _channel_vector(lm.last_activations[target], None)

    # build a subgraph: source (cut) -> target
    sess, in_name, out_name, ext_inputs = _build_subgraph_session(
        lm, source, target)
    ext = _ext_feed(lm, ext_inputs, source)

    src = np.array(lm.last_activations[source], dtype=np.float32)

    def perturb(a):
        a = a.copy()
        if source_channel is None:
            a *= (strength if mode == "boost" else 0.0)
        else:
            if a.ndim == 4:
                c = min(source_channel, a.shape[1] - 1)
                a[0, c] = a[0, c] * (strength if mode == "boost" else 0.0)
            elif a.ndim == 3:
                c = min(source_channel, a.shape[2] - 1)
                a[0, :, c] = a[0, :, c] * (strength if mode == "boost" else 0.0)
            elif a.ndim == 2:
                c = min(source_channel, a.shape[1] - 1)
                a[0, c] = a[0, c] * (strength if mode == "boost" else 0.0)
        return a

    feed = dict(ext); feed[in_name] = perturb(src)
    new_target = _channel_vector(sess.run([out_name], feed)[0], None)

    n = min(len(base_target), len(new_target))
    delta = new_target[:n] - base_target[:n]
    order = np.argsort(-np.abs(delta))[:top]
    rows = [{"channel": int(i), "delta": _finite(delta[i]),
             "base": _finite(base_target[i]), "new": _finite(new_target[i])}
            for i in order]
    total = float(np.abs(delta).sum())
    return {"source": source, "target": target,
            "source_channel": source_channel, "mode": mode,
            "strength": strength, "n_target": int(n),
            "total_change": _finite(total),
            "mean_abs_change": _finite(np.abs(delta).mean()),
            "top": rows}


def circuit_ablate(lm: LoadedModel, source: str, source_channel: int | None,
                   top: int = 8) -> dict:
    """Zero the source unit and measure the impact on the model's final output
    — how much the model's prediction relies on this unit."""
    if source not in lm.last_activations:
        raise KeyError("Run a stimulus first, then ablate.")
    logits_name = _logits_output(lm)
    if _depth_of(lm, source) >= _depth_of(lm, logits_name) - 0:
        # source at/after logits: nothing downstream
        pass
    base_out = np.array(lm.last_activations.get(logits_name), dtype=np.float32)
    if base_out is None or base_out.size == 0:
        raise ValueError("No final output recorded to measure impact against.")
    base_flat = base_out.reshape(base_out.shape[0], -1)[0] if base_out.ndim >= 2 \
        else base_out.reshape(-1)

    sess, in_name, out_name, ext_inputs = _build_subgraph_session(
        lm, source, logits_name)
    ext = _ext_feed(lm, ext_inputs, source)
    src = np.array(lm.last_activations[source], dtype=np.float32)
    a = src.copy()
    if source_channel is None:
        a *= 0.0
    else:
        if a.ndim == 4:
            a[0, min(source_channel, a.shape[1]-1)] = 0.0
        elif a.ndim == 3:
            a[0, :, min(source_channel, a.shape[2]-1)] = 0.0
        elif a.ndim == 2:
            a[0, min(source_channel, a.shape[1]-1)] = 0.0
    feed = dict(ext); feed[in_name] = a
    new_out = sess.run([out_name], feed)[0]
    new_flat = new_out.reshape(new_out.shape[0], -1)[0] if new_out.ndim >= 2 \
        else new_out.reshape(-1)

    n = min(len(base_flat), len(new_flat))
    delta = new_flat[:n] - base_flat[:n]

    def label(i):
        if lm.labels and i < len(lm.labels):
            return lm.labels[i]
        if lm.tokenizer:
            return lm.tokenizer.id_to_token(int(i))
        return f"#{i}"
    order = np.argsort(-np.abs(delta))[:top]
    rows = [{"index": int(i), "label": label(int(i)),
             "delta": _finite(delta[i]), "base": _finite(base_flat[i]),
             "new": _finite(new_flat[i])} for i in order]
    # did the argmax prediction change?
    pred_before = int(np.argmax(base_flat))
    pred_after = int(np.argmax(new_flat))
    return {"source": source, "source_channel": source_channel,
            "n_out": int(n), "l2_change": _finite(np.linalg.norm(delta)),
            "pred_changed": pred_before != pred_after,
            "pred_before": {"index": pred_before, "label": label(pred_before)},
            "pred_after": {"index": pred_after, "label": label(pred_after)},
            "top": rows}


def circuit_targets(lm: LoadedModel, source: str) -> dict:
    """List candidate target layers (deeper than source) for the trace UI."""
    sd = _depth_of(lm, source)
    outs = []
    for n in lm.graph["nodes"]:
        a = lm.last_activations.get(n["id"])
        if not isinstance(a, np.ndarray) or a.dtype.kind != "f":
            continue
        if n["depth"] > sd:
            outs.append({"id": n["id"], "op": n["op"], "depth": n["depth"]})
    outs.sort(key=lambda x: x["depth"])
    return {"source": source, "source_depth": sd, "targets": outs}

# ---------------------------------------------------------------------------
# Model diffing — compare two models on the same stimulus
#   Replays model A's exact input tensors through model B and compares per-layer
#   activations. Layers are matched by node id (base vs fine-tuned share ids),
#   falling back to (depth, op) matching for renamed graphs. Surfaces where the
#   two networks diverge — i.e. what a fine-tune / edit actually changed.
# ---------------------------------------------------------------------------

def _match_nodes(a: LoadedModel, b: LoadedModel) -> list[tuple[str, str, str, int]]:
    """Pairs of (node_id_a, node_id_b, op, depth) present in both graphs."""
    b_by_id = {n["id"]: n for n in b.graph["nodes"]}
    b_by_key: dict[tuple, list] = {}
    for n in b.graph["nodes"]:
        b_by_key.setdefault((n["depth"], n["op"]), []).append(n)
    pairs, used = [], set()
    # 1) exact id match
    for n in a.graph["nodes"]:
        if n["id"] in b_by_id and n["id"] not in used:
            pairs.append((n["id"], n["id"], n["op"], n["depth"]))
            used.add(n["id"])
    if pairs:
        return pairs
    # 2) fallback: match by (depth, op), in order
    for n in a.graph["nodes"]:
        cands = b_by_key.get((n["depth"], n["op"]), [])
        for cand in cands:
            if cand["id"] not in used:
                pairs.append((n["id"], cand["id"], n["op"], n["depth"]))
                used.add(cand["id"])
                break
    return pairs


def _feed_for_b(a: LoadedModel, b: LoadedModel) -> dict:
    """Adapt model A's recorded input tensors to model B's input names."""
    if not a.last_feed:
        raise KeyError("Run a stimulus on the base model first.")
    b_inputs = {i["name"]: i for i in b.inputs}
    feed = {}
    # direct name matches
    for name, val in a.last_feed.items():
        if name in b_inputs:
            feed[name] = np.asarray(val)
    # for any B input still unfilled, try positional match by shape/rank
    missing = [i for i in b.inputs if i["name"] not in feed]
    leftovers = [v for k, v in a.last_feed.items() if k not in feed]
    for inp in missing:
        for j, v in enumerate(leftovers):
            if np.asarray(v).ndim == len(inp["shape"]):
                feed[inp["name"]] = np.asarray(v)
                leftovers.pop(j)
                break
    # fill any still-missing companion inputs (masks etc.)
    _fill_companions(b, feed)
    for inp in b.inputs:                       # last resort: zeros
        if inp["name"] not in feed:
            dt = np.dtype(inp["dtype"])
            feed[inp["name"]] = np.zeros(inp["shape"], dtype=dt)
    return feed


def model_diff(a: LoadedModel, b: LoadedModel, top: int = 20) -> dict:
    """Run A's stimulus through both models; compare per-layer activations."""
    if not a.last_feed:
        raise KeyError("Run a stimulus on the base model (A) first.")
    pairs = _match_nodes(a, b)
    if not pairs:
        raise ValueError("These two models share no matchable layers — diffing "
                         "needs the same (or closely related) architecture.")
    # ensure A activations are current
    acts_a = a.last_activations or {}
    feed_b = _feed_for_b(a, b)
    outs_b = b.session.run(b.output_names, feed_b)
    acts_b = dict(zip(b.output_names, outs_b))

    rows = []
    for aid, bid, op, depth in pairs:
        ta, tb = acts_a.get(aid), acts_b.get(bid)
        if not isinstance(ta, np.ndarray) or not isinstance(tb, np.ndarray):
            continue
        if ta.dtype.kind not in "fiu" or tb.dtype.kind not in "fiu":
            continue
        if ta.size == 0 or tb.size == 0:
            continue
        fa = _sanitize_array(ta.astype(np.float32)).reshape(-1)
        fb = _sanitize_array(tb.astype(np.float32)).reshape(-1)
        n = min(fa.size, fb.size)
        if n == 0:
            continue
        fa, fb = fa[:n], fb[:n]
        diff = fb - fa
        denom = (np.linalg.norm(fa) + np.linalg.norm(fb)) / 2 + 1e-8
        rel = float(np.linalg.norm(diff) / denom)          # relative L2 change
        # cosine similarity of the two activation vectors
        cos = float(np.dot(fa, fb) /
                    (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-8))
        rows.append({"node_a": aid, "node_b": bid, "op": op, "depth": depth,
                     "rel_l2": _finite(rel), "cosine": _finite(cos),
                     "mean_abs_a": _finite(np.abs(fa).mean()),
                     "mean_abs_b": _finite(np.abs(fb).mean()),
                     "size": int(n)})
    rows.sort(key=lambda r: r["depth"])
    ranked = sorted(rows, key=lambda r: -r["rel_l2"])[:top]
    # output-level comparison
    logits_name = _logits_output(a)
    out_div = None
    if logits_name in acts_a and logits_name in acts_b:
        la = _sanitize_array(np.asarray(acts_a[logits_name], np.float32)).reshape(-1)
        lb = _sanitize_array(np.asarray(acts_b[logits_name], np.float32)).reshape(-1)
        m = min(la.size, lb.size)
        if m:
            pa, pb = int(np.argmax(la[:m])), int(np.argmax(lb[:m]))

            def lab(i):
                if a.labels and i < len(a.labels):
                    return a.labels[i]
                if a.tokenizer:
                    return a.tokenizer.id_to_token(int(i))
                return f"#{i}"
            out_div = {"pred_a": {"index": pa, "label": lab(pa)},
                       "pred_b": {"index": pb, "label": lab(pb)},
                       "pred_changed": pa != pb,
                       "rel_l2": _finite(np.linalg.norm(lb[:m]-la[:m]) /
                                         ((np.linalg.norm(la[:m])+np.linalg.norm(lb[:m]))/2+1e-8))}
    return {"n_matched": len(rows), "match_mode":
            "id" if pairs and pairs[0][0] == pairs[0][1] else "depth-op",
            "profile": [{"depth": r["depth"], "op": r["op"],
                         "rel_l2": r["rel_l2"], "cosine": r["cosine"],
                         "node_a": r["node_a"]} for r in rows],
            "top_divergent": ranked, "output": out_div}

# ---------------------------------------------------------------------------
# Activation maximization — synthesize the input a unit most wants to see
#
# Occlusion asks "which part of THIS input matters?"; ranking asks "which of my
# inputs fires it hardest?". Maximization asks the neuron directly: it searches
# input space for the stimulus that drives the unit as hard as possible.
#
# ONNX graphs aren't autodiff-friendly, so instead of gradient ascent this uses
# an NES-style evolutionary search: sample a population of perturbations, score
# each with a forward pass, and step along the fitness-weighted direction. For
# images the search is regularized with blur + jitter — without it the optimizer
# finds adversarial high-frequency noise that maximizes the unit but shows no
# structure. Integer (token) inputs get discrete hill-climbing instead.
# ---------------------------------------------------------------------------

AMAX_MAX_STEPS = 1200


def _blur_axes(x: np.ndarray):
    """The two spatial axes of an image-like tensor, or None."""
    if x.ndim == 4:
        if x.shape[1] in (1, 3):
            return (2, 3)                      # NCHW
        if x.shape[3] in (1, 3):
            return (1, 2)                      # NHWC
    return None


def _box_blur(x: np.ndarray, axes, r: int = 1) -> np.ndarray:
    k = 2 * r + 1
    out = np.zeros_like(x)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out += np.roll(np.roll(x, dy, axis=axes[0]), dx, axis=axes[1])
    return out / (k * k)


def _vocab_size(lm: LoadedModel) -> int:
    if lm.tokenizer is not None and hasattr(lm.tokenizer, "vocab_size"):
        try:
            return max(2, int(lm.tokenizer.vocab_size()))
        except Exception:
            pass
    try:
        name = _logits_output(lm)
        o = next((o for o in lm.graph["outputs"] if o["name"] == name), None)
        sh = [d for d in (o["shape"] if o else []) if isinstance(d, int) and d > 0]
        if sh and sh[-1] > 1:
            return int(sh[-1])
    except Exception:
        pass
    return 256


def _maximize_float(lm, node, channel, prim, steps, pop, sigma, lr,
                    regularize, rng, t0, budget_s):
    import time
    shape = prim["shape"]
    x = (rng.standard_normal(shape) * 0.08).astype(np.float32)
    axes = _blur_axes(x) if regularize else None

    def fit(v, jitter=False):
        vv = v
        if axes is not None and jitter:              # translation robustness
            vv = np.roll(np.roll(v, int(rng.integers(-2, 3)), axis=axes[0]),
                         int(rng.integers(-2, 3)), axis=axes[1])
        feed = {prim["name"]: vv.astype(np.float32)}
        _fill_companions(lm, feed)
        return _target_scalar(_run_feed_get(lm, feed, node), channel)

    start = fit(x)
    best_x, best = x.copy(), start
    trace = [_finite(start)]
    ran = 0
    for s in range(steps):
        if time.time() - t0 > budget_s:
            break
        noise = rng.standard_normal((pop, *shape)).astype(np.float32)
        fits = np.array([fit(x + sigma * noise[i], jitter=True)
                         for i in range(pop)], np.float32)
        adv = fits - fits.mean()
        adv /= (fits.std() + 1e-8)                   # normalized advantages
        grad = (adv[:, None] * noise.reshape(pop, -1)).mean(axis=0).reshape(shape)
        x = x + lr * sigma * grad
        if axes is not None and s % 3 == 2:          # kill high-frequency junk
            x = _box_blur(x, axes, 1)
        rms = float(np.linalg.norm(x)) / max(1.0, math.sqrt(x.size))
        if rms > 1.5:                                # keep it image-scaled
            x *= 1.5 / rms
        cur = fit(x)
        trace.append(_finite(cur))
        if cur > best:
            best, best_x = cur, x.copy()
        ran = s + 1

    feed = {prim["name"]: best_x.astype(np.float32)}
    _fill_companions(lm, feed)
    run = run_stimulus(lm, feed)                      # whole UI now reflects it
    lm.last_stimulus = {prim["name"]: {
        "mode": "maximize", "modality": prim.get("modality", "tensor"),
        "text": None, "has_blob": False, "sample_rate": 16000,
        "_normalize": "amax"}}
    return {"kind": "float", "node": node, "channel": channel,
            "start": _finite(start), "best": _finite(best),
            "gain": _finite(best / (abs(start) + 1e-8)),
            "steps_run": ran, "trace": trace,
            "regularized": axes is not None, "run": run}


def _maximize_tokens(lm, node, channel, prim, steps, rng, t0, budget_s,
                     cand: int = 8):
    """Discrete hill-climbing over token ids."""
    import time
    shape = prim["shape"]
    dt = np.dtype(prim["dtype"])
    T = int(shape[-1])
    V = _vocab_size(lm)
    ids = rng.integers(0, V, size=T)

    def fit(vec):
        x = np.asarray(vec, dtype=dt).reshape(shape)
        feed = {prim["name"]: x}
        _fill_companions(lm, feed)
        return _target_scalar(_run_feed_get(lm, feed, node), channel)

    start = fit(ids)
    best = start
    trace = [_finite(start)]
    ran = 0
    for s in range(steps):
        if time.time() - t0 > budget_s:
            break
        pos = int(rng.integers(0, T))
        base = ids[pos]
        picks = rng.integers(0, V, size=cand)
        improved = False
        for p in picks:
            ids[pos] = p
            f = fit(ids)
            if f > best:
                best, base, improved = f, p, True
        ids[pos] = base                              # keep the winner
        trace.append(_finite(best))
        ran = s + 1

    x = np.asarray(ids, dtype=dt).reshape(shape)
    feed = {prim["name"]: x}
    _fill_companions(lm, feed)
    run = run_stimulus(lm, feed)
    toks = [{"id": int(i), "token": (lm.tokenizer.id_to_token(int(i))
                                     if lm.tokenizer else f"#{i}")} for i in ids]
    lm.last_tokens = toks
    run["tokens"] = toks
    lm.last_stimulus = {prim["name"]: {
        "mode": "maximize", "modality": "text", "text": None,
        "has_blob": False, "sample_rate": 16000, "_normalize": "unit"}}
    return {"kind": "tokens", "node": node, "channel": channel,
            "start": _finite(start), "best": _finite(best),
            "gain": _finite(best / (abs(start) + 1e-8)),
            "steps_run": ran, "trace": trace,
            "tokens": [t["token"] for t in toks], "run": run}


def activation_maximize(lm: LoadedModel, node: str, channel: int | None = None,
                        steps: int = 240, pop: int = 16, sigma: float = 0.8,
                        lr: float = 1.5, regularize: bool = True,
                        seed: int | None = None,
                        budget_s: float = 30.0) -> dict:
    import time
    if node not in {n["id"] for n in lm.graph["nodes"]}:
        raise ValueError(f"Unknown layer '{node}'.")
    prim = next((i for i in lm.inputs if i["name"] == lm.primary_input), None)
    if prim is None:
        raise ValueError("No primary input to optimize.")
    if any(not isinstance(d, int) or d <= 0 for d in prim["shape"]):
        raise ValueError("Input shape isn't fully resolved — apply shapes first.")
    steps = max(4, min(int(steps), AMAX_MAX_STEPS))
    pop = max(4, min(int(pop), 32))
    rng = np.random.default_rng(0 if seed is None else int(seed))
    t0 = time.time()
    dt = np.dtype(prim["dtype"])
    if dt.kind in "iu":
        return _maximize_tokens(lm, node, channel, prim, steps, rng, t0, budget_s)
    return _maximize_float(lm, node, channel, prim, steps, pop, sigma, lr,
                           regularize, rng, t0, budget_s)

# ---------------------------------------------------------------------------
# Network health scan — is the model wasting capacity?
#
# Everything else in AIFmri is single-stimulus. This runs a batch of varied
# stimuli and aggregates per-unit statistics across them, which is the only way
# to see the pathologies that matter in practice:
#   dead      — the unit never fires for ANY stimulus (classic dead ReLU)
#   weak      — it fires, but negligibly next to its layer-mates
#   constant  — it never varies across stimuli: carries no information
#   duplicate — two units whose responses are near-perfectly correlated, i.e.
#               the layer is narrower than it looks
# ---------------------------------------------------------------------------

HEALTH_MAX_UNITS = 512
# On a 1300-layer model, keeping a per-unit signature for every node at once
# costs hundreds of MB and OOM'd the slow tests. Sample evenly across depth
# instead: the point is to find pathological LAYERS, and an even sample across
# the network finds them just as well.
HEALTH_MAX_LAYERS = 120

# Pure data-movement ops have no units of their own — they're views of the
# previous layer, so "dead units" there is a meaningless restatement of the
# layer before. Skip them so the report only covers layers that compute.
HEALTH_SKIP_OPS = {
    "Flatten", "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Identity",
    "Cast", "Shape", "Constant", "ConstantOfShape", "Slice", "Pad", "Concat",
    "Split", "Expand", "Tile", "Gather" if False else "DequantizeLinear",
}


def _unit_samples(a: np.ndarray, k: int = 12) -> np.ndarray:
    """(units, k) — a sample of each unit's activation PATTERN, not just its
    magnitude. Duplicate detection needs the pattern: two channels with similar
    average energy are not duplicates, two with the same spatial map are."""
    x = np.asarray(a, dtype=np.float32)
    if x.ndim == 4:                       # (B,C,H,W) -> unit = channel
        m = x[0].reshape(x.shape[1], -1)
    elif x.ndim == 3:                     # (B,T,D) -> unit = feature dim
        m = x[0].T
    elif x.ndim == 2:                     # (B,D) -> one value per unit
        m = x[0][:, None]
    else:
        m = x.reshape(-1)[:, None]
    if m.shape[1] > k:
        m = m[:, np.linspace(0, m.shape[1] - 1, k).astype(int)]
    return m


def _random_stimulus(lm: LoadedModel, prim: dict, rng, scale: float = 1.0):
    dt = np.dtype(prim["dtype"])
    if dt.kind == "f":
        x = (rng.standard_normal(prim["shape"]) * scale).astype(np.float32)
    else:
        x = rng.integers(0, _vocab_size(lm), size=prim["shape"]).astype(dt)
    feed = {prim["name"]: x}
    _fill_companions(lm, feed)
    return feed


def _dup_pairs(S: np.ndarray, thresh: float, cap: int = 256):
    """Near-duplicate unit pairs from an (n_units, n_features) signature matrix,
    where each row is one unit's activation pattern across all probe stimuli."""
    if S.shape[1] < 3 or S.shape[0] < 2:
        return 0, []
    X = S[:cap]
    sd = X.std(axis=1)
    live = np.where(sd > 1e-8)[0]                # constant rows break corrcoef
    if live.size < 2:
        return 0, []
    C = np.corrcoef(X[live], rowvar=True)
    C = np.nan_to_num(C, nan=0.0)
    iu = np.triu_indices(C.shape[0], k=1)
    vals = C[iu]
    hits = np.where(vals >= thresh)[0]
    pairs = [[int(live[iu[0][h]]), int(live[iu[1][h]]), _finite(vals[h])]
             for h in hits[:12]]
    return int(hits.size), pairs


def health_scan(lm: LoadedModel, n: int = 24, seed: int | None = None,
                dup_thresh: float = 0.98, budget_s: float = 45.0) -> dict:
    import time
    t0 = time.time()
    prim = next((i for i in lm.inputs if i["name"] == lm.primary_input), None)
    if prim is None:
        raise ValueError("No primary input to probe.")
    if any(not isinstance(d, int) or d <= 0 for d in prim["shape"]):
        raise ValueError("Input shape isn't fully resolved — apply shapes first.")
    n = max(4, min(int(n), 64))
    rng = np.random.default_rng(0 if seed is None else int(seed))
    computing = [x for x in lm.graph["nodes"]
                 if x.get("op") not in HEALTH_SKIP_OPS]
    if len(computing) > HEALTH_MAX_LAYERS:      # even sample across depth
        computing.sort(key=lambda x: x["depth"])
        idx = np.linspace(0, len(computing) - 1, HEALTH_MAX_LAYERS).astype(int)
        computing = [computing[i] for i in sorted(set(idx.tolist()))]
    node_ids = [x["id"] for x in computing]
    acc: dict[str, list] = {}      # per-unit mean|act| per stimulus (dead/weak)
    sig: dict[str, list] = {}      # per-unit activation pattern  (duplicates)
    runs = 0
    for s in range(n):
        if time.time() - t0 > budget_s:
            break
        feed = _random_stimulus(lm, prim, rng, scale=float(rng.uniform(0.5, 2.0)))
        outs = lm.session.run(lm.output_names, feed)
        acts = dict(zip(lm.output_names, outs))
        for nid in node_ids:
            a = acts.get(nid)
            if not isinstance(a, np.ndarray) or a.dtype.kind not in "fiu":
                continue
            if a.size == 0:
                continue
            a = _sanitize_array(a.astype(np.float32))
            v = _channel_vector(a, None)
            keep = None
            if v.size > HEALTH_MAX_UNITS:
                keep = np.linspace(0, v.size - 1, HEALTH_MAX_UNITS).astype(int)
                v = v[keep]
            acc.setdefault(nid, []).append(v)
            p = _unit_samples(a)
            if keep is not None and p.shape[0] > keep.max():
                p = p[keep]
            sig.setdefault(nid, []).append(p)
        runs += 1
    if runs < 3:
        raise ValueError("Not enough probe runs completed to judge health.")

    by_id = {x["id"]: x for x in lm.graph["nodes"]}
    rows, tot = [], {"units": 0, "dead": 0, "weak": 0, "constant": 0, "dup": 0}
    for nid, vecs in acc.items():
        L = min(len(v) for v in vecs)
        M = np.stack([v[:L] for v in vecs])          # (runs, units)
        if M.shape[1] < 1:
            continue
        peak = M.max(axis=0)
        sd = M.std(axis=0)
        scale = float(np.percentile(peak, 95)) or 1.0
        dead = peak <= 1e-7                          # never fires, at all
        weak = (~dead) & (peak < 0.01 * scale)       # fires negligibly
        const = (~dead) & (sd <= 1e-6)               # never varies: no info
        S = sig.get(nid)
        if S:
            w = min(x.shape[0] for x in S)
            S = np.concatenate([x[:w] for x in S], axis=1)   # (units, runs*k)
            ndup, pairs = _dup_pairs(S, dup_thresh)
        else:
            ndup, pairs = 0, []
        node = by_id.get(nid, {})
        u = int(M.shape[1])
        rows.append({
            "node": nid, "op": node.get("op", "?"), "depth": node.get("depth", 0),
            "units": u,
            "dead": int(dead.sum()), "dead_pct": _finite(dead.mean() * 100),
            "weak": int(weak.sum()), "constant": int(const.sum()),
            "dup_pairs": ndup, "dup_examples": pairs,
            "peak": _finite(peak.max()), "mean_sd": _finite(sd.mean()),
        })
        tot["units"] += u
        tot["dead"] += int(dead.sum())
        tot["weak"] += int(weak.sum())
        tot["constant"] += int(const.sum())
        tot["dup"] += ndup
    rows.sort(key=lambda r: r["depth"])
    worst = sorted(rows, key=lambda r: (-r["dead_pct"], -r["dup_pairs"]))[:12]
    tot["dead_pct"] = _finite(tot["dead"] / max(tot["units"], 1) * 100)
    tot["runs"] = runs
    tot["layers"] = len(rows)
    return {"totals": tot, "profile": rows, "worst": worst,
            "dup_thresh": dup_thresh}

# ---------------------------------------------------------------------------
# Weight / filter viewer — what the network KNOWS (vs what it does)
#
# Every other view here is about activations: the network responding. Weights
# are the other half — the learned structure itself. First-layer conv kernels
# rendered as image tiles are the oldest trick in interpretability and still
# the most legible one; deeper layers get norms and distributions instead,
# since a 3x3x512 kernel has no honest 2-D picture.
# ---------------------------------------------------------------------------

def _proto(lm: LoadedModel):
    if getattr(lm, "_proto_cache", None) is None:
        p = onnx.load(str(lm.onnx_path))
        lm._proto_cache = shape_inference.infer_shapes(p)
    return lm._proto_cache


def _node_weights(lm: LoadedModel, node: str):
    """The initializer tensors feeding a node: [(name, ndarray), ...]."""
    from onnx import numpy_helper
    g = _proto(lm).graph
    inits = {i.name: i for i in g.initializer}
    prod = None
    for nd in g.node:
        if node in nd.output:
            prod = nd
            break
    if prod is None:
        raise KeyError(f"'{node}' isn't produced by any node.")
    out = []
    for inp in prod.input:
        if inp in inits:
            out.append((inp, numpy_helper.to_array(inits[inp])))
    return prod.op_type, out


def weight_info(lm: LoadedModel, node: str, bins: int = 28) -> dict:
    """Stats + histogram for a layer's learned parameters."""
    op, ws = _node_weights(lm, node)
    if not ws:
        return {"node": node, "op": op, "has_weights": False,
                "note": "This layer has no learned parameters of its own — "
                        "it's an activation, pooling or shape op."}
    tensors = []
    for name, a in ws:
        a = np.asarray(a, dtype=np.float32)
        flat = _sanitize_array(a).ravel()
        lo, hi = float(flat.min()), float(flat.max())
        hist, edges = np.histogram(flat, bins=bins, range=(lo, hi) if hi > lo
                                   else (lo - 1e-3, lo + 1e-3))
        # per-filter norms: for conv (O,I,kh,kw) / gemm (O,I) the first axis
        # indexes output units, so this lines up with the channel selector.
        norms = None
        if a.ndim >= 2:
            norms = np.linalg.norm(a.reshape(a.shape[0], -1), axis=1)
            if norms.size > 512:
                norms = norms[np.linspace(0, norms.size - 1, 512).astype(int)]
            norms = [_finite(v) for v in norms]
        tensors.append({
            "name": name, "shape": list(a.shape), "size": int(a.size),
            "kind": "kernel" if a.ndim == 4 else
                    ("matrix" if a.ndim == 2 else "bias" if a.ndim == 1 else "tensor"),
            "min": _finite(lo), "max": _finite(hi),
            "mean": _finite(flat.mean()), "std": _finite(flat.std()),
            "l2": _finite(np.linalg.norm(flat)),
            "zeros_pct": _finite((np.abs(flat) < 1e-8).mean() * 100),
            "hist": hist.tolist(), "edges": [_finite(lo), _finite(hi)],
            "filter_norms": norms,
        })
    viewable = any(t["kind"] == "kernel" and
                   next(w for n, w in ws if n == t["name"]).shape[1] in (1, 3)
                   for t in tensors)
    return {"node": node, "op": op, "has_weights": True,
            "tensors": tensors, "viewable_kernels": viewable}


def weight_image(lm: LoadedModel, node: str, cols: int = 8,
                 upscale: int = 18) -> bytes:
    """Render conv kernels as a tiled contact sheet.

    RGB kernels (in_channels == 3) render in colour — that's the classic
    first-layer view. Deeper kernels have many input channels and no honest
    colour mapping, so each filter is shown as its mean-over-input-channels
    map on the inferno scale instead."""
    op, ws = _node_weights(lm, node)
    kern = next((a for _, a in ws if np.asarray(a).ndim == 4), None)
    if kern is None:
        raise ValueError("This layer has no 4-D convolution kernel to render.")
    W = _sanitize_array(np.asarray(kern, dtype=np.float32))     # (O, I, kh, kw)
    O, I, kh, kw = W.shape
    n = min(O, 64)
    cols = max(1, min(int(cols), 16))
    rows = int(math.ceil(n / cols))
    pad = 1
    colour = (I == 3)
    tile_h, tile_w = kh, kw
    sheet = np.zeros((rows * (tile_h + pad) + pad,
                      cols * (tile_w + pad) + pad, 3), np.float32)
    for f in range(n):
        k = W[f]
        if colour:
            t = k.transpose(1, 2, 0)                    # (kh,kw,3)
            lo, hi = float(t.min()), float(t.max())
            t = (t - lo) / (hi - lo + 1e-8)
        else:
            m = np.abs(k).mean(axis=0)                  # (kh,kw)
            lo, hi = float(m.min()), float(m.max())
            m = (m - lo) / (hi - lo + 1e-8)
            t = np.stack([np.vectorize(lambda v: inferno_rgb(v)[c])(m)
                          for c in range(3)], axis=-1)
        r, c = divmod(f, cols)
        y = pad + r * (tile_h + pad)
        x = pad + c * (tile_w + pad)
        sheet[y:y + tile_h, x:x + tile_w] = t
    # nearest-neighbour upscale so 3x3 kernels are actually visible
    up = max(1, min(int(upscale), 48))
    sheet = np.repeat(np.repeat(sheet, up, axis=0), up, axis=1)
    return _to_png(np.clip(sheet, 0, 1))


_INF_STOPS = [(0.0, (0.0, 0.0, 0.02)), (0.25, (0.28, 0.05, 0.36)),
              (0.5, (0.63, 0.18, 0.35)), (0.75, (0.93, 0.44, 0.15)),
              (1.0, (0.99, 0.91, 0.60))]


def inferno_rgb(t: float):
    t = max(0.0, min(1.0, float(t)))
    for i in range(len(_INF_STOPS) - 1):
        a, ca = _INF_STOPS[i]
        b, cb = _INF_STOPS[i + 1]
        if t <= b:
            f = (t - a) / (b - a + 1e-9)
            return tuple(ca[j] + (cb[j] - ca[j]) * f for j in range(3))
    return _INF_STOPS[-1][1]

# ---------------------------------------------------------------------------
# Latency lens — paint the brain by TIME instead of activation
#
# ONNX Runtime can emit a per-node profile. We run the model a few times with
# profiling on, parse the trace, and attribute microseconds to graph nodes. The
# same 3-D view then answers a different question: not "what lit up?" but
# "where does the time actually go?"
# ---------------------------------------------------------------------------

def latency_profile(lm: LoadedModel, runs: int = 8,
                    warmup: int = 2) -> dict:
    import json as _json
    import os as _os
    import tempfile as _tf
    import onnxruntime as ort

    feed = lm.last_feed
    if not feed:
        prim = next((i for i in lm.inputs if i["name"] == lm.primary_input), None)
        if prim is None or any(not isinstance(d, int) or d <= 0
                               for d in prim["shape"]):
            raise ValueError("Run a stimulus first so there's an input to time.")
        rng = np.random.default_rng(0)
        feed = _random_stimulus(lm, prim, rng)

    runs = max(2, min(int(runs), 32))
    tmpdir = _tf.mkdtemp(prefix="aifmri_prof_")
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = _os.path.join(tmpdir, "prof")
    try:
        sess = ort.InferenceSession(str(lm.onnx_path), so,
                                    providers=["CPUExecutionProvider"])
    except Exception as e:
        raise ValueError(f"Could not open a profiling session: {e}")
    names = [o.name for o in sess.get_outputs()]
    # only feed what this (unexposed) session actually declares
    want = {i.name for i in sess.get_inputs()}
    f2 = {k: v for k, v in feed.items() if k in want}
    for i in sess.get_inputs():
        if i.name not in f2:
            raise ValueError(f"Missing input '{i.name}' for the timing run.")
    for _ in range(warmup):
        sess.run(names, f2)
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(names, f2)
    wall = (time.perf_counter() - t0) / runs * 1000.0
    path = sess.end_profiling()

    per_node: dict[str, list] = {}
    try:
        with open(path) as fh:
            events = _json.load(fh)
    except Exception as e:
        raise ValueError(f"Could not read the ORT profile: {e}")
    finally:
        try:
            _os.remove(path)
            _os.rmdir(tmpdir)
        except Exception:
            pass

    ops_seen: dict[str, str] = {}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("cat") != "Node" or "dur" not in ev:
            continue
        args = ev.get("args") or {}
        if args.get("op_name") is None:
            continue
        nm = ev.get("name", "")
        # ORT names node events "<node_name>_kernel_time", and rewrites node
        # names during graph optimization: layout transforms add "_nchwc",
        # fusion collapses e.g. Conv+Relu into one kernel named after the LAST
        # node in the fused run. Strip the decorations, then match.
        base = nm.rsplit("_kernel_time", 1)[0]
        for suf in ("_nchwc", "_fence_before", "_fence_after", "_token"):
            if base.endswith(suf):
                base = base[: -len(suf)]
        per_node.setdefault(base, []).append(float(ev["dur"]))
        ops_seen[base] = args.get("op_name")

    if not per_node:
        raise ValueError("The profiler returned no per-node timings.")

    # map ORT node names -> our graph nodes (our id is the node's first output)
    g = _proto(lm).graph
    by_name, by_out = {}, {}
    for nd in g.node:
        out0 = nd.output[0] if nd.output else None
        if not out0:
            continue
        if nd.name:
            by_name[nd.name] = out0
        by_out[out0] = out0

    rows = []
    depth = {n["id"]: n["depth"] for n in lm.graph["nodes"]}
    op_of = {n["id"]: n["op"] for n in lm.graph["nodes"]}
    agg: dict[str, float] = {}
    fused: dict[str, str] = {}
    overhead_us, overhead_ops = 0.0, []
    for base, durs in per_node.items():
        med = float(np.median(durs))
        nid = by_name.get(base) or (base if base in by_out else None)
        if nid is None:
            # ORT-internal work with no counterpart in our graph (layout
            # reorders, memcpy). Real time — report it, don't silently drop it.
            overhead_us += med
            overhead_ops.append(ops_seen.get(base, base))
            continue
        agg[nid] = agg.get(nid, 0.0) + med
        ran_op = ops_seen.get(base)
        if ran_op and op_of.get(nid) and ran_op != op_of.get(nid):
            fused[nid] = ran_op          # ORT fused: kernel op != graph op
    total = (sum(agg.values()) + overhead_us) or 1.0
    for nid, us in agg.items():
        rows.append({"node": nid, "op": op_of.get(nid, "?"),
                     "depth": depth.get(nid, 0), "us": _finite(us),
                     "pct": _finite(us / total * 100),
                     "fused_as": fused.get(nid)})
    rows.sort(key=lambda r: r["depth"])
    top = sorted(rows, key=lambda r: -r["us"])[:12]
    mx = max((r["us"] for r in rows), default=1.0) or 1.0
    return {"runs": runs, "wall_ms": _finite(wall),
            "total_us": _finite(total), "matched": len(rows),
            "overhead_us": _finite(overhead_us),
            "overhead_ops": sorted(set(overhead_ops))[:6],
            "fused": {k: v for k, v in fused.items()},
            "max_us": _finite(mx), "profile": rows, "top": top}

# ---------------------------------------------------------------------------
# Session export — a self-contained HTML report
#
# Every insight in AIFmri currently dies when the tab closes. This snapshots
# the current session — model, stimulus, activation profile, and whichever
# analyses have been run — into one standalone HTML file with the images
# inlined as data URIs. No server, no assets: openable anywhere, e-mailable.
# ---------------------------------------------------------------------------

def _b64_png(data: bytes) -> str:
    import base64
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def export_report(lm: LoadedModel, notes: str = "",
                  findings: dict | None = None) -> bytes:
    import datetime
    findings = findings or {}
    acts = lm.last_activations or {}
    rows = []
    for n in sorted(lm.graph["nodes"], key=lambda x: x["depth"]):
        a = acts.get(n["id"])
        if not isinstance(a, np.ndarray) or a.dtype.kind not in "fiu":
            continue
        if a.size == 0:
            continue
        f = _sanitize_array(a.astype(np.float32))
        rows.append((n, _finite(np.abs(f).mean()), _finite(f.std()),
                     list(f.shape)))
    mx = max((r[1] for r in rows), default=1.0) or 1.0

    stim_html = ""
    try:
        png = stimulus_image(lm)
        stim_html = (f'<img class="stim" src="{_b64_png(png)}" '
                     f'alt="stimulus">')
    except Exception:
        if lm.last_tokens:
            chips = "".join(f'<span class="tok">{_esc(t["token"])}</span>'
                            for t in lm.last_tokens[:64])
            stim_html = f'<div class="toks">{chips}</div>'
        else:
            stim_html = '<div class="muted">no viewable stimulus</div>'

    bars = "".join(
        f'<tr><td class="mono">d{n["depth"]}</td><td class="mono">{_esc(n["op"])}</td>'
        f'<td class="mono small">{_esc(n["id"])[:34]}</td>'
        f'<td class="mono small">{"×".join(str(s) for s in shp)}</td>'
        f'<td class="barcell"><div class="bar" style="width:{max(1, e/mx*100):.1f}%"></div></td>'
        f'<td class="mono">{e:.4f}</td></tr>'
        for n, e, sd, shp in rows)

    extra = ""
    for title, body in findings.items():
        extra += f'<h2>{_esc(title)}</h2>{body}'

    when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>AIFmri report — {_esc(lm.source_name or lm.model_id)}</title>
<style>
:root{{--bg:#070B14;--panel:#0B1220;--line:#1A2436;--fg:#E6EDF7;--muted:#7C8CA6;--hot:#FFB020;--mono:ui-monospace,'SF Mono',Menlo,monospace}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;padding:32px}}
.wrap{{max-width:1000px;margin:0 auto}}
h1{{font-size:20px;letter-spacing:.14em;margin:0 0 4px}}
h1 b{{color:var(--hot)}}
h2{{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);
   border-bottom:1px solid var(--line);padding-bottom:6px;margin:28px 0 12px}}
.mono{{font-family:var(--mono)}} .small{{font-size:11px}} .muted{{color:var(--muted)}}
.meta{{color:var(--muted);font-family:var(--mono);font-size:11px;margin-bottom:18px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:14px;margin-bottom:12px}}
table{{width:100%;border-collapse:collapse}}
td,th{{padding:3px 8px;text-align:left;border-bottom:1px solid #101828;font-size:12px}}
.barcell{{width:46%}}
.bar{{height:7px;background:linear-gradient(90deg,#3B0F70,#F1605D,#FEC287);border-radius:2px}}
.stim{{max-width:220px;image-rendering:pixelated;border:1px solid var(--line);border-radius:4px}}
.toks span.tok{{display:inline-block;background:#101A2B;border:1px solid var(--line);
  padding:1px 5px;margin:2px;border-radius:3px;font-family:var(--mono);font-size:11px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.k{{color:var(--muted);font-family:var(--mono);font-size:11px}}
.v{{color:var(--hot);font-family:var(--mono);font-size:16px}}
.note{{white-space:pre-wrap;font-family:var(--mono);font-size:12px;color:var(--fg)}}
footer{{margin-top:28px;color:var(--muted);font-size:11px;font-family:var(--mono)}}
</style></head><body><div class="wrap">
<h1>AI<b>FMRI</b> — session report</h1>
<div class="meta">{_esc(lm.source_name or lm.model_id)} · {len(lm.graph["nodes"])} layers ·
 depth {lm.graph.get("depth_max", 0)} · exported {when}</div>

{f'<h2>Notes</h2><div class="card note">{_esc(notes)}</div>' if notes.strip() else ''}

<h2>Stimulus</h2>
<div class="card">{stim_html}</div>

<h2>Activation profile</h2>
<div class="card"><table>
<tr><th>depth</th><th>op</th><th>layer</th><th>shape</th><th>mean |activation|</th><th></th></tr>
{bars}
</table></div>
{extra}
<footer>Generated by AIFmri — functional imaging for neural networks.
Activations captured from a single stimulus; this file is self-contained.</footer>
</div></body></html>"""
    return html.encode("utf-8")
