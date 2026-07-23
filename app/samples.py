"""Built-in sample models, generated on first request (pure ONNX, no torch).

  demo-cnn          3x32x32 image classifier (conv/relu/pool x2 + fc)
  demo-transformer  byte-level single-block transformer LM, dynamic seq len
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper as h

rng = np.random.default_rng(7)


def _w(name, shape, scale=0.1):
    return h.make_tensor(name, T.FLOAT, shape,
                         (rng.standard_normal(shape) * scale)
                         .astype(np.float32).ravel())


def build_cnn(path: Path):
    nodes = [
        h.make_node("Conv", ["input", "w1", "b1"], ["conv1"], name="conv1",
                    kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        h.make_node("Relu", ["conv1"], ["relu1"], name="relu1"),
        h.make_node("MaxPool", ["relu1"], ["pool1"], name="pool1",
                    kernel_shape=[2, 2], strides=[2, 2]),
        h.make_node("Conv", ["pool1", "w2", "b2"], ["conv2"], name="conv2",
                    kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        h.make_node("Relu", ["conv2"], ["relu2"], name="relu2"),
        h.make_node("MaxPool", ["relu2"], ["pool2"], name="pool2",
                    kernel_shape=[2, 2], strides=[2, 2]),
        h.make_node("Flatten", ["pool2"], ["flat"], name="flatten"),
        h.make_node("Gemm", ["flat", "wf", "bf"], ["logits"], name="fc",
                    transB=1),
    ]
    inits = [_w("w1", [16, 3, 3, 3]), _w("b1", [16]),
             _w("w2", [32, 16, 3, 3]), _w("b2", [32]),
             _w("wf", [10, 32 * 8 * 8]), _w("bf", [10])]
    g = h.make_graph(nodes, "demo_cnn",
                     [h.make_tensor_value_info("input", T.FLOAT, [1, 3, 32, 32])],
                     [h.make_tensor_value_info("logits", T.FLOAT, [1, 10])],
                     inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 10
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def build_transformer(path: Path, vocab=259, d=32, ffn=64):
    """One attention block + FFN over byte tokens; dynamic sequence length."""
    sqrt_d = h.make_tensor("sqrt_d", T.FLOAT, [], [float(np.sqrt(d))])
    nodes = [
        h.make_node("Gather", ["emb_table", "input_ids"], ["emb"], name="embed"),
        h.make_node("MatMul", ["emb", "Wq"], ["q"], name="attn_q"),
        h.make_node("MatMul", ["emb", "Wk"], ["k"], name="attn_k"),
        h.make_node("MatMul", ["emb", "Wv"], ["v"], name="attn_v"),
        h.make_node("Transpose", ["k"], ["kT"], name="attn_kT", perm=[0, 2, 1]),
        h.make_node("MatMul", ["q", "kT"], ["scores_raw"], name="attn_scores"),
        h.make_node("Div", ["scores_raw", "sqrt_d"], ["scores"], name="attn_scale"),
        h.make_node("Softmax", ["scores"], ["attn"], name="attn_softmax", axis=-1),
        h.make_node("MatMul", ["attn", "v"], ["ctx"], name="attn_ctx"),
        h.make_node("Add", ["emb", "ctx"], ["res1"], name="residual_1"),
        h.make_node("MatMul", ["res1", "W1"], ["ffn_pre"], name="ffn_in"),
        h.make_node("Relu", ["ffn_pre"], ["ffn_act"], name="ffn_relu"),
        h.make_node("MatMul", ["ffn_act", "W2"], ["ffn_out"], name="ffn_out"),
        h.make_node("Add", ["res1", "ffn_out"], ["res2"], name="residual_2"),
        h.make_node("MatMul", ["res2", "Wout"], ["logits"], name="lm_head"),
    ]
    inits = [_w("emb_table", [vocab, d], 0.3), _w("Wq", [d, d], 0.3),
             _w("Wk", [d, d], 0.3), _w("Wv", [d, d], 0.3),
             _w("W1", [d, ffn], 0.3), _w("W2", [ffn, d], 0.3),
             _w("Wout", [d, vocab], 0.3), sqrt_d]
    g = h.make_graph(nodes, "demo_transformer",
                     [h.make_tensor_value_info("input_ids", T.INT64,
                                               [1, "seq_len"])],
                     [h.make_tensor_value_info("logits", T.FLOAT,
                                               [1, "seq_len", vocab])],
                     inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 10
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def build_audio(path: Path, n_mels=64, n_frames=256):
    """Conv net over a log-mel spectrogram — input named so it's detected
    as audio: (1, 1, mel, frames)."""
    nodes = [
        h.make_node("Conv", ["mel_spectrogram", "aw1", "ab1"], ["aconv1"],
                    name="conv1", kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        h.make_node("Relu", ["aconv1"], ["arelu1"], name="relu1"),
        h.make_node("MaxPool", ["arelu1"], ["apool1"], name="pool1",
                    kernel_shape=[4, 4], strides=[4, 4]),
        h.make_node("Conv", ["apool1", "aw2", "ab2"], ["aconv2"],
                    name="conv2", kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        h.make_node("Relu", ["aconv2"], ["arelu2"], name="relu2"),
        h.make_node("GlobalAveragePool", ["arelu2"], ["agap"], name="gap"),
        h.make_node("Flatten", ["agap"], ["aflat"], name="flatten"),
        h.make_node("Gemm", ["aflat", "awf", "abf"], ["alogits"], name="fc",
                    transB=1),
    ]
    inits = [_w("aw1", [16, 1, 3, 3]), _w("ab1", [16]),
             _w("aw2", [32, 16, 3, 3]), _w("ab2", [32]),
             _w("awf", [8, 32]), _w("abf", [8])]
    g = h.make_graph(nodes, "demo_audio",
                     [h.make_tensor_value_info("mel_spectrogram", T.FLOAT,
                                               [1, 1, n_mels, n_frames])],
                     [h.make_tensor_value_info("alogits", T.FLOAT, [1, 8])],
                     inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 10
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def build_video(path: Path, t=8, hw=32):
    """3-D conv net over sampled frames — input (1, 3, T, H, W)."""
    nodes = [
        h.make_node("Conv", ["video", "vw1", "vb1"], ["vconv1"], name="conv3d_1",
                    kernel_shape=[3, 3, 3], pads=[1, 1, 1, 1, 1, 1]),
        h.make_node("Relu", ["vconv1"], ["vrelu1"], name="relu1"),
        h.make_node("MaxPool", ["vrelu1"], ["vpool1"], name="pool1",
                    kernel_shape=[2, 2, 2], strides=[2, 2, 2]),
        h.make_node("Conv", ["vpool1", "vw2", "vb2"], ["vconv2"], name="conv3d_2",
                    kernel_shape=[3, 3, 3], pads=[1, 1, 1, 1, 1, 1]),
        h.make_node("Relu", ["vconv2"], ["vrelu2"], name="relu2"),
        h.make_node("GlobalAveragePool", ["vrelu2"], ["vgap"], name="gap"),
        h.make_node("Flatten", ["vgap"], ["vflat"], name="flatten"),
        h.make_node("Gemm", ["vflat", "vwf", "vbf"], ["vlogits"], name="fc",
                    transB=1),
    ]
    inits = [_w("vw1", [8, 3, 3, 3, 3]), _w("vb1", [8]),
             _w("vw2", [16, 8, 3, 3, 3]), _w("vb2", [16]),
             _w("vwf", [6, 16]), _w("vbf", [6])]
    g = h.make_graph(nodes, "demo_video",
                     [h.make_tensor_value_info("video", T.FLOAT,
                                               [1, 3, t, hw, hw])],
                     [h.make_tensor_value_info("vlogits", T.FLOAT, [1, 6])],
                     inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 10
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def build_clip(path: Path, vocab=259, d=16, hw=32):
    """Two-tower CLIP-style model: image tower + byte-text tower project
    into a shared d-dim space; output is their similarity plus both
    embeddings. Inputs: pixel_values (1,3,hw,hw) and input_ids (1,seq)."""
    nodes = [
        # image tower
        h.make_node("Conv", ["pixel_values", "iw1", "ib1"], ["img_conv"],
                    name="img_conv", kernel_shape=[3, 3], strides=[2, 2],
                    pads=[1, 1, 1, 1]),
        h.make_node("Relu", ["img_conv"], ["img_relu"], name="img_relu"),
        h.make_node("GlobalAveragePool", ["img_relu"], ["img_gap"],
                    name="img_gap"),
        h.make_node("Flatten", ["img_gap"], ["img_flat"], name="img_flatten"),
        h.make_node("MatMul", ["img_flat", "iproj"], ["img_emb"],
                    name="img_proj"),
        # text tower
        h.make_node("Gather", ["emb_table", "input_ids"], ["txt_tok"],
                    name="txt_embed"),
        h.make_node("ReduceMean", ["txt_tok"], ["txt_pool"], name="txt_pool",
                    axes=[1], keepdims=0),
        h.make_node("MatMul", ["txt_pool", "tproj"], ["txt_emb"],
                    name="txt_proj"),
        # cross-modal similarity
        h.make_node("Transpose", ["txt_emb"], ["txt_embT"], name="txt_T",
                    perm=[1, 0]),
        h.make_node("MatMul", ["img_emb", "txt_embT"], ["similarity"],
                    name="similarity"),
    ]
    inits = [_w("iw1", [8, 3, 3, 3], 0.3), _w("ib1", [8], 0.1),
             _w("iproj", [8, d], 0.4), _w("emb_table", [vocab, d], 0.4),
             _w("tproj", [d, d], 0.4)]
    g = h.make_graph(nodes, "demo_clip",
                     [h.make_tensor_value_info("pixel_values", T.FLOAT,
                                               [1, 3, hw, hw]),
                      h.make_tensor_value_info("input_ids", T.INT64,
                                               [1, "seq_len"])],
                     [h.make_tensor_value_info("similarity", T.FLOAT, [1, 1]),
                      h.make_tensor_value_info("img_emb", T.FLOAT, [1, d]),
                      h.make_tensor_value_info("txt_emb", T.FLOAT, [1, d])],
                     inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 10
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


SAMPLES = {
    "demo-cnn": {
        "title": "Demo CNN — image classifier",
        "desc": "3×32×32 conv net. Try an image or noise stimulus.",
        "build": build_cnn,
    },
    "demo-transformer": {
        "title": "Demo transformer — byte-level LM",
        "desc": "One attention block over UTF-8 bytes. Type a text stimulus "
                "and watch the attention light up.",
        "build": build_transformer,
    },
    "demo-audio": {
        "title": "Demo audio net — mel spectrogram",
        "desc": "Conv net over 64-mel log spectrograms. Upload a sound or "
                "record from your mic.",
        "build": build_audio,
    },
    "demo-video": {
        "title": "Demo video net — 3-D conv",
        "desc": "Conv3D over 8 sampled frames. Drop any short video clip.",
        "build": build_video,
    },
    "demo-clip": {
        "title": "Demo CLIP — image × text",
        "desc": "Two towers meet in a similarity score. Give it an image "
                "AND a caption, watch both towers fire.",
        "build": build_clip,
    },
}


def ensure_sample(name: str, workdir: Path) -> Path:
    if name not in SAMPLES:
        raise KeyError(f"No sample called '{name}'.")
    path = workdir / f"{name}.onnx"
    if not path.exists():
        SAMPLES[name]["build"](path)
    return path
