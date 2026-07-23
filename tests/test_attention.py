"""Attention detection and extraction.

Detection is structural (a Softmax whose score chain traces back to a QK^T
MatMul), so the precision tests matter: CLIP's similarity MatMul looks like
attention and must NOT be flagged.
"""

import numpy as np
import pytest


def test_attention_detected_on_the_transformer(client):
    d = client.post("/api/samples/demo-transformer").json()
    assert any(n.get("attention") for n in d["graph"]["nodes"])


@pytest.mark.parametrize("sample", ["demo-cnn", "demo-audio", "demo-video",
                                    "demo-clip"])
def test_no_false_positive_attention(client, sample):
    """demo-clip is the trap: it has a similarity MatMul, but no softmax
    over QK^T, so it is not attention."""
    d = client.post(f"/api/samples/{sample}").json()
    assert not any(n.get("attention") for n in d["graph"]["nodes"])


def test_attention_rows_are_a_probability_distribution(client, transformer):
    a = client.get(f"/api/models/{transformer}/attention",
                   params={"node": "attn"}).json()
    M = np.array(a["matrix"])
    assert M.shape[0] == a["q"] and M.shape[1] == a["k"]
    assert np.allclose(M.sum(axis=1), 1.0, atol=0.02)


def test_attention_matrix_matches_token_count(client, transformer):
    a = client.get(f"/api/models/{transformer}/attention",
                   params={"node": "attn"}).json()
    assert a["q"] == len("the cat sat on the mat")


def test_attention_needs_a_stimulus(client):
    mid = client.post("/api/samples/demo-transformer").json()["model_id"]
    r = client.get(f"/api/models/{mid}/attention", params={"node": "attn"})
    assert r.status_code == 409


def test_fused_attention_is_flagged_and_falls_back(client):
    """Flash/fused kernels never materialize the QK^T matrix. We must flag the
    op and degrade to per-token output energy rather than fake a heatmap."""
    import numpy as np
    import onnx
    from onnx import helper as h, TensorProto as T

    d, vocab = 16, 259
    nodes = [
        h.make_node("Gather", ["emb_table", "input_ids"], ["emb"], name="embed"),
        h.make_node("Attention", ["emb", "qkv_w", "qkv_b"], ["attn_out"],
                    name="fused_attn", domain="com.microsoft", num_heads=2),
        h.make_node("MatMul", ["attn_out", "Wout"], ["logits"], name="head"),
    ]
    rng = np.random.default_rng(0)
    inits = [
        h.make_tensor("emb_table", T.FLOAT, [vocab, d],
                      (rng.standard_normal((vocab, d)) * .3).astype(np.float32).ravel()),
        h.make_tensor("qkv_w", T.FLOAT, [d, 3 * d],
                      (rng.standard_normal((d, 3 * d)) * .3).astype(np.float32).ravel()),
        h.make_tensor("qkv_b", T.FLOAT, [3 * d], np.zeros(3 * d, np.float32)),
        h.make_tensor("Wout", T.FLOAT, [d, vocab],
                      (rng.standard_normal((d, vocab)) * .3).astype(np.float32).ravel()),
    ]
    g = h.make_graph(nodes, "fused",
                     [h.make_tensor_value_info("input_ids", T.INT64, [1, "seq"])],
                     [h.make_tensor_value_info("logits", T.FLOAT, [1, "seq", vocab])],
                     inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17),
                                       h.make_opsetid("com.microsoft", 1)])
    m.ir_version = 10

    d0 = client.post("/api/models",
                     files={"file": ("fused.onnx", m.SerializeToString())}).json()
    if d0.get("status") != "ok":
        pytest.skip("fused Attention op unsupported by this onnxruntime build")
    node = next(n for n in d0["graph"]["nodes"] if n.get("attention"))
    assert node.get("attention_fused") is True

    mid = d0["model_id"]
    client.post(f"/api/models/{mid}/tokenizer", data={"kind": "byte"})
    client.post(f"/api/models/{mid}/run", data={"mode": "text", "text": "flash"})
    a = client.get(f"/api/models/{mid}/attention",
                   params={"node": node["id"]}).json()
    assert a["fused"] is True
    assert "token_energy" in a and "matrix" not in a
