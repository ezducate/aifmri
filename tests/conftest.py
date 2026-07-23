"""Shared fixtures.

Models are session-scoped: loading and running them dominates the runtime, and
none of the tests mutate the graph, so we pay that cost once.
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient          # noqa: E402
from app.main import app                           # noqa: E402
import app.core as core                            # noqa: E402


DIGITS = [f"digit_{i}" for i in range(10)]


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def noise_image() -> bytes:
    rng = np.random.default_rng(0)
    return _png((rng.random((32, 32, 3)) * 255).astype(np.uint8))


@pytest.fixture(scope="session")
def square_image() -> bytes:
    """Black 32x32 with a bright square in the TOP-LEFT quadrant.

    Ground truth for occlusion: importance must peak inside that square.
    """
    img = np.zeros((32, 32, 3), np.uint8)
    img[4:14, 4:14] = 255
    return _png(img)


@pytest.fixture(scope="session")
def cnn(client, noise_image):
    """demo-cnn with class labels attached and a noise image already run."""
    mid = client.post("/api/samples/demo-cnn").json()["model_id"]
    client.post(f"/api/models/{mid}/labels",
                files={"file": ("l.json", json.dumps(DIGITS).encode())})
    r = client.post(f"/api/models/{mid}/run",
                    data={"mode": "image", "normalize": "unit"},
                    files={"image": ("x.png", noise_image)})
    assert r.status_code == 200
    return mid


@pytest.fixture(scope="session")
def transformer(client):
    """demo-transformer with the byte tokenizer and a sentence already run."""
    mid = client.post("/api/samples/demo-transformer").json()["model_id"]
    client.post(f"/api/models/{mid}/tokenizer", data={"kind": "byte"})
    r = client.post(f"/api/models/{mid}/run",
                    data={"mode": "text", "text": "the cat sat on the mat"})
    assert r.status_code == 200
    return mid


@pytest.fixture(scope="session")
def conv1_kernels(client, cnn) -> np.ndarray:
    """The real conv1 weights, read straight off the ONNX file."""
    import onnx
    from onnx import numpy_helper
    proto = onnx.load(str(core.MODELS[cnn].onnx_path))
    for init in proto.graph.initializer:
        a = numpy_helper.to_array(init)
        if a.ndim == 4:
            return a
    pytest.skip("no conv kernels found")


def build_sabotaged(client, dead=(0, 1, 2), dup=(5, 6)) -> str:
    """A demo-cnn clone with KNOWN pathologies, for ground-truth testing:
    conv1 channels in `dead` can never fire (huge negative bias), and the two
    channels in `dup` are made byte-identical. Returns the loaded model_id."""
    import onnx
    from onnx import numpy_helper
    base = client.post("/api/samples/demo-cnn").json()["model_id"]
    proto = onnx.load(str(core.MODELS[base].onnx_path))
    for init in proto.graph.initializer:
        a = numpy_helper.to_array(init).copy()
        if a.ndim == 4:                         # conv1 weights
            a[dup[1]] = a[dup[0]]
            init.CopyFrom(numpy_helper.from_array(a, init.name))
        elif a.ndim == 1 and a.shape[0] >= 8:   # conv1 bias
            a[list(dead)] = -1000.0
            a[dup[1]] = a[dup[0]]
            init.CopyFrom(numpy_helper.from_array(a, init.name))
            break
    blob = proto.SerializeToString()
    return client.post("/api/models",
                       files={"file": ("sabotaged.onnx", blob)}).json()["model_id"]


def build_head_perturbed(client, delta=0.5) -> str:
    """A demo-cnn clone with ONLY the classifier head changed. Every layer
    before it must be bit-identical — the diff test depends on that."""
    import onnx
    from onnx import numpy_helper
    base = client.post("/api/samples/demo-cnn").json()["model_id"]
    proto = onnx.load(str(core.MODELS[base].onnx_path))
    for init in proto.graph.initializer:
        a = numpy_helper.to_array(init)
        if a.ndim == 2:                         # the Gemm weight
            init.CopyFrom(numpy_helper.from_array(a + delta, init.name))
            break
    blob = proto.SerializeToString()
    return client.post("/api/models",
                       files={"file": ("head.onnx", blob)}).json()["model_id"]
