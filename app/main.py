"""AIFmri — FastAPI entrypoint (v0.2).

Run:  uvicorn app.main:app --reload
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import core, samples

from .version import __version__

app = FastAPI(title="AIFmri", version=__version__)

STATIC = Path(__file__).parent / "static"
WORKDIR = Path(tempfile.gettempdir()) / "aifmri_models"
WORKDIR.mkdir(exist_ok=True)


def _get(model_id: str) -> core.LoadedModel:
    lm = core.MODELS.get(model_id)
    if lm is None:
        raise HTTPException(404, "Unknown model_id — load a model first, or "
                                 "it was evicted to free memory.")
    core.touch_model(model_id)          # keep the LRU order honest
    return lm


def _load_response(result) -> dict:
    if "needs_arch" in result:                      # state_dict recovery flow
        return {"status": "needs_arch",
                "pending_id": result["pending_id"],
                "n_tensors": result["n_tensors"],
                "candidates": result["candidates"],
                "message": ("This .pt file is a bare state_dict — weights "
                            "without an architecture. Pick a matching "
                            "architecture below, or paste the model class "
                            "code it belongs to.")}
    lm, probe = result["model"], result["probe"]
    return {"status": "ok", "model_id": lm.model_id, "graph": lm.graph,
            "inputs": lm.inputs, "primary_input": lm.primary_input,
            "stimulus_inputs": [i["name"] for i in core.stimulus_inputs(lm)],
            "source_name": lm.source_name, "probe": probe,
            "diffusion": core.is_diffusion(lm)}


@app.post("/api/models")
async def upload_model(file: UploadFile = File(...),
                       input_shape: str | None = Form(None)):
    suffix = Path(file.filename or "model.onnx").suffix or ".onnx"
    dest = WORKDIR / f"upload_{Path(tempfile.mktemp()).name}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    shape = json.loads(input_shape) if input_shape else None
    try:
        result = core.load_model(dest, shape, source_name=file.filename or "")
    except Exception as e:
        raise HTTPException(422, str(e))
    return _load_response(result)


@app.post("/api/models/resolve")
async def resolve_state_dict(pending_id: str = Form(...),
                             arch: str | None = Form(None),
                             code: str | None = Form(None),
                             input_shape: str | None = Form(None)):
    shape = json.loads(input_shape) if input_shape else None
    try:
        result = core.resolve_pending(pending_id, arch, code, shape)
    except Exception as e:
        raise HTTPException(422, f"Recovery failed: {e}")
    return _load_response(result)


@app.get("/api/samples")
def list_samples():
    return {"samples": [{"name": k, "title": v["title"], "desc": v["desc"]}
                        for k, v in samples.SAMPLES.items()]}


@app.post("/api/samples/{name}")
def load_sample(name: str):
    try:
        path = samples.ensure_sample(name, WORKDIR)
        result = core.load_model(path, source_name=name)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(422, str(e))
    return _load_response(result)


@app.post("/api/models/{model_id}/shapes")
async def set_shapes(model_id: str, shapes: str = Form(...)):
    lm = _get(model_id)
    try:
        probe = core.apply_shapes(lm, json.loads(shapes))
    except Exception as e:
        raise HTTPException(422, f"Bad shapes: {e}")
    return {"inputs": lm.inputs, "probe": probe}


@app.post("/api/models/{model_id}/run")
async def run_stimulus(model_id: str,
                       mode: str = Form("noise"),
                       seed: int | None = Form(None),
                       normalize: str = Form("unit"),
                       values: str | None = Form(None),
                       text: str | None = Form(None),
                       sample_rate: int = Form(16000),
                       image: UploadFile | None = File(None),
                       media: UploadFile | None = File(None)):
    lm = _get(model_id)
    try:
        up = media or image
        blob = await up.read() if up is not None else None
        vals = json.loads(values) if values else None
        feed = core.make_feed(lm, mode, blob, vals, text, seed, normalize,
                              sample_rate)
        return core.run_stimulus(lm, feed)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"Stimulus failed: {e}")


@app.post("/api/models/{model_id}/run_multi")
async def run_multi(model_id: str, request: Request):
    """Multimodal stimulus: form field `stimuli` is a JSON object
    {input_name: {mode, text?, normalize?, sample_rate?, seed?}}; media
    files are attached under keys `media__<input_name>`."""
    lm = _get(model_id)
    form = await request.form()
    try:
        specs = json.loads(form.get("stimuli") or "{}")
        for key in form:
            if key.startswith("media__"):
                up = form[key]
                if hasattr(up, "read"):
                    specs.setdefault(key[len("media__"):], {})["_blob"] = \
                        await up.read()
        feed = core.make_multi_feed(lm, specs)
        return core.run_stimulus(lm, feed)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"Stimulus failed: {e}")


@app.post("/api/models/{model_id}/tokenizer")
async def attach_tokenizer(model_id: str,
                           kind: str = Form("byte"),
                           file: UploadFile | None = File(None)):
    lm = _get(model_id)
    if kind == "byte":
        lm.tokenizer = core.ByteTokenizer()
        return {"kind": "byte", "note": "UTF-8 bytes as token ids."}
    if kind == "hf":
        if file is None:
            raise HTTPException(422, "Upload a tokenizer.json for kind=hf.")
        try:
            lm.tokenizer = core.HFTokenizer(await file.read())
        except Exception as e:
            raise HTTPException(422, f"Could not parse tokenizer.json: {e}")
        return {"kind": "hf"}
    raise HTTPException(422, "kind must be 'byte' or 'hf'.")


@app.get("/api/models/{model_id}/raw")
def get_raw(model_id: str, node: str, channel: int | None = None,
            offset: int = 0, limit: int = Query(256, le=core.RAW_SLICE_MAX)):
    try:
        return core.raw_slice(_get(model_id), node, channel, offset, limit)
    except KeyError as e:
        raise HTTPException(409, str(e))


@app.get("/api/models/{model_id}/spatial")
def get_spatial(model_id: str, node: str, channel: int | None = None):
    try:
        return core.spatial_for(_get(model_id), node, channel)
    except KeyError as e:
        raise HTTPException(409, str(e))


@app.get("/api/models/{model_id}/decode/image")
def decode_image(model_id: str, node: str):
    try:
        png = core.decode_as_image(_get(model_id), node)
    except (KeyError, ValueError) as e:
        raise HTTPException(409, str(e))
    return Response(content=png, media_type="image/png")


@app.get("/api/models/{model_id}/decode/topk")
def decode_topk(model_id: str, node: str, k: int = 10):
    try:
        return core.decode_topk(_get(model_id), node, k)
    except KeyError as e:
        raise HTTPException(409, str(e))


@app.post("/api/models/{model_id}/labels")
async def attach_labels(model_id: str, file: UploadFile = File(...)):
    lm = _get(model_id)
    data = json.loads(await file.read())
    if not isinstance(data, list):
        raise HTTPException(422, "Labels file must be a JSON array of strings.")
    lm.labels = [str(x) for x in data]
    return {"count": len(lm.labels)}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")

@app.get("/api/models/{model_id}/stats")
def get_stats(model_id: str, node: str, bins: int = Query(28, le=96)):
    try:
        return core.node_detail(_get(model_id), node, bins)
    except KeyError as e:
        raise HTTPException(409, str(e))


@app.post("/api/models/{model_id}/record")
async def record_sequence(model_id: str,
                          mode: str = Form("noise"),
                          frames: int = Form(48),
                          text: str | None = Form(None),
                          sample_rate: int = Form(16000),
                          normalize: str = Form("unit"),
                          seed: int | None = Form(None),
                          schedule: str = Form("scaled_linear"),
                          media: UploadFile | None = File(None)):
    """Run a stimulus sequence and record a per-layer activation timeline."""
    lm = _get(model_id)
    try:
        blob = await media.read() if media is not None else None
        return core.start_recording(lm, mode, frames, blob, text,
                                    sample_rate, normalize, seed, schedule)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"Recording failed: {e}")


@app.post("/api/models/{model_id}/record/seek")
async def record_seek(model_id: str, frame: int = Form(...)):
    """Re-run one recorded frame at full fidelity (units + inspector cache)."""
    lm = _get(model_id)
    try:
        return core.seek_frame(lm, frame)
    except KeyError as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        raise HTTPException(422, f"Seek failed: {e}")

@app.get("/api/models/{model_id}/attention")
def get_attention(model_id: str, node: str, head: int | None = None,
                  max_tok: int = Query(64, le=128)):
    lm = _get(model_id)
    try:
        return core.attention_map(lm, node, head, max_tok)
    except KeyError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))

from . import hf_gallery

HF_CACHE = WORKDIR / "hf_cache"


@app.get("/api/hf")
def list_hf():
    ok, hint = hf_gallery.exporter_available()
    return {"models": hf_gallery.gallery_list(), "available": ok, "hint": hint}


@app.post("/api/hf/{name}")
async def load_hf(name: str):
    """Export (or reuse) a HuggingFace gallery model and load it."""
    try:
        info = hf_gallery.export_hf(name, HF_CACHE)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(422, str(e))
    try:
        result = core.load_model(info["onnx_path"], source_name=name)
    except Exception as e:
        raise HTTPException(422, f"Loaded export but could not open it: {e}")
    resp = _load_response(result)
    meta = info["meta"]
    # auto-attach tokenizer for text models
    if info["tokenizer_path"] and result.get("model"):
        try:
            result["model"].tokenizer = core.HFTokenizer(
                info["tokenizer_path"].read_bytes())
            resp["tokenizer_attached"] = True
        except Exception:
            resp["tokenizer_attached"] = False
    resp["hf"] = {"default_text": meta.get("default_text"),
                  "normalize": meta.get("normalize"),
                  "modality": meta.get("modality"),
                  "trained": meta.get("trained", False),
                  "schedule": meta.get("schedule")}
    return resp

# ---- Stimulus viewer ----

@app.get("/api/models/{model_id}/stimulus")
def stimulus_info(model_id: str):
    return core.stimulus_info(_get(model_id))


@app.get("/api/models/{model_id}/stimulus/image")
def stimulus_image(model_id: str, input: str | None = None,
                   frame: int | None = None):
    lm = _get(model_id)
    try:
        png = core.stimulus_image(lm, input, frame)
    except (KeyError, ValueError) as e:
        raise HTTPException(409, str(e))
    return Response(content=png, media_type="image/png")


@app.get("/api/models/{model_id}/stimulus/waveform")
def stimulus_waveform(model_id: str, input: str | None = None):
    lm = _get(model_id)
    try:
        return core.stimulus_waveform(lm, input)
    except (KeyError, ValueError) as e:
        raise HTTPException(409, str(e))


# ---- Jacobian lens ----

@app.get("/api/models/{model_id}/jlens")
def jlens(model_id: str, node: str, position: int | None = None,
          k: int = 12, n_probe: int = 48):
    lm = _get(model_id)
    try:
        return core.jacobian_lens(lm, node, position, k, n_probe)
    except KeyError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/models/{model_id}/jlens/stack")
def jlens_stack(model_id: str, k: int = 5, n_probe: int = 32,
                position: int | None = None,
                max_layers: int = Query(14, ge=2, le=64)):
    lm = _get(model_id)
    try:
        return core.jacobian_lens_stack(lm, k, n_probe, position, max_layers)
    except KeyError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))

# ---- Neuron attribution ----

@app.get("/api/models/{model_id}/attribution/occlusion")
def attribution_occlusion(model_id: str, node: str,
                          channel: int | None = None,
                          input: str | None = None,
                          grid: int = Query(10, le=14),
                          frame: int | None = None):
    lm = _get(model_id)
    try:
        return core.attribution_occlusion(lm, node, channel, input, grid, frame)
    except KeyError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/models/{model_id}/attribution/rank_frames")
def attribution_rank_frames(model_id: str, node: str,
                            channel: int | None = None,
                            top: int = Query(8, le=32)):
    lm = _get(model_id)
    try:
        return core.attribution_rank_frames(lm, node, channel, top)
    except KeyError as e:
        raise HTTPException(409, str(e))


@app.get("/api/models/{model_id}/attribution/rank_noise")
def attribution_rank_noise(model_id: str, node: str,
                           channel: int | None = None,
                           n: int = Query(24, le=64),
                           top: int = Query(6, le=16)):
    lm = _get(model_id)
    try:
        return core.attribution_rank_noise(lm, node, channel, n, top)
    except (KeyError, ValueError) as e:
        raise HTTPException(409, str(e))

# ---- Circuit tracing ----

@app.get("/api/models/{model_id}/circuit/targets")
def circuit_targets(model_id: str, source: str):
    lm = _get(model_id)
    try:
        return core.circuit_targets(lm, source)
    except KeyError as e:
        raise HTTPException(409, str(e))


@app.get("/api/models/{model_id}/circuit/trace")
def circuit_trace(model_id: str, source: str, target: str,
                  source_channel: int | None = None, mode: str = "boost",
                  strength: float = 2.0, top: int = Query(12, le=48)):
    lm = _get(model_id)
    try:
        return core.circuit_trace(lm, source, target, source_channel, mode,
                                  strength, top)
    except KeyError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/models/{model_id}/circuit/ablate")
def circuit_ablate(model_id: str, source: str,
                   source_channel: int | None = None,
                   top: int = Query(8, le=32)):
    lm = _get(model_id)
    try:
        return core.circuit_ablate(lm, source, source_channel, top)
    except KeyError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))

# ---- Model diffing ----

@app.get("/api/diff")
def model_diff(model_a: str, model_b: str, top: int = Query(20, le=64)):
    a, b = _get(model_a), _get(model_b)
    try:
        return core.model_diff(a, b, top)
    except KeyError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/models")
def list_models():
    """Loaded models available to diff against."""
    return {"models": [{"model_id": mid, "source_name": lm.source_name or mid,
                        "n_nodes": len(lm.graph["nodes"])}
                       for mid, lm in core.MODELS.items()]}

@app.post("/api/models/{model_id}/attribution/maximize")
def attribution_maximize(model_id: str, node: str = Form(...),
                         channel: int | None = Form(None),
                         steps: int = Form(240), pop: int = Form(16),
                         sigma: float = Form(0.8), lr: float = Form(1.5),
                         regularize: bool = Form(True),
                         seed: int | None = Form(None)):
    """Synthesize the input that maximally excites a unit (gradient-free)."""
    lm = _get(model_id)
    try:
        return core.activation_maximize(lm, node, channel, steps, pop, sigma,
                                        lr, regularize, seed)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(422, f"Maximization failed: {e}")

@app.get("/api/models/{model_id}/health")
def health_scan(model_id: str, n: int = Query(24, le=64),
                seed: int | None = None,
                dup_thresh: float = Query(0.98, ge=0.5, le=1.0)):
    """Probe the model with varied stimuli; report dead/weak/duplicate units."""
    lm = _get(model_id)
    try:
        return core.health_scan(lm, n, seed, dup_thresh)
    except ValueError as e:
        raise HTTPException(422, str(e))

# ---- Weight / filter viewer ----

@app.get("/api/models/{model_id}/weights")
def weights(model_id: str, node: str, bins: int = Query(28, le=96)):
    lm = _get(model_id)
    try:
        return core.weight_info(lm, node, bins)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(422, f"Could not read weights: {e}")


@app.get("/api/models/{model_id}/weights/image")
def weights_image(model_id: str, node: str, cols: int = Query(8, le=16),
                  upscale: int = Query(18, le=48)):
    lm = _get(model_id)
    try:
        png = core.weight_image(lm, node, cols, upscale)
    except (KeyError, ValueError) as e:
        raise HTTPException(409, str(e))
    return Response(content=png, media_type="image/png")

# ---- Latency lens ----

@app.get("/api/models/{model_id}/latency")
def latency(model_id: str, runs: int = Query(8, le=32)):
    lm = _get(model_id)
    try:
        return core.latency_profile(lm, runs)
    except ValueError as e:
        raise HTTPException(422, str(e))

# ---- Session export ----

@app.post("/api/models/{model_id}/export")
def export_report(model_id: str, notes: str = Form(""),
                  include_health: bool = Form(False),
                  include_latency: bool = Form(False)):
    """A self-contained HTML report of the current session."""
    lm = _get(model_id)
    findings = {}
    if include_health:
        try:
            h = core.health_scan(lm, n=16)
            t = h["totals"]
            bad = [w for w in h["worst"] if w["dead"] or w["dup_pairs"]][:8]
            body = (f'<div class="card"><div class="grid">'
                    f'<div><div class="k">units probed</div><div class="v">{t["units"]}</div></div>'
                    f'<div><div class="k">never fire</div><div class="v">{t["dead"]} '
                    f'({t["dead_pct"]:.1f}%)</div></div></div>')
            if bad:
                body += '<table><tr><th>layer</th><th>dead</th><th>dup pairs</th></tr>'
                body += "".join(f'<tr><td class="mono small">d{w["depth"]} {core._esc(w["op"])}</td>'
                                f'<td class="mono">{w["dead"]}</td>'
                                f'<td class="mono">{w["dup_pairs"]}</td></tr>' for w in bad)
                body += '</table>'
            findings["Network health"] = body + '</div>'
        except Exception as e:
            findings["Network health"] = f'<div class="card muted">unavailable: {core._esc(e)}</div>'
    if include_latency:
        try:
            l = core.latency_profile(lm, runs=6)
            mx = l["max_us"] or 1
            body = (f'<div class="card"><div class="grid">'
                    f'<div><div class="k">per inference</div>'
                    f'<div class="v">{l["wall_ms"]:.3f} ms</div></div>'
                    f'<div><div class="k">bottleneck</div><div class="v">'
                    f'{core._esc(l["top"][0]["op"]) if l["top"] else "—"}</div></div></div>'
                    '<table><tr><th>layer</th><th>time</th><th>share</th><th></th></tr>')
            body += "".join(
                f'<tr><td class="mono small">d{t["depth"]} {core._esc(t["op"])}</td>'
                f'<td class="mono">{t["us"]:.1f} µs</td><td class="mono">{t["pct"]:.1f}%</td>'
                f'<td class="barcell"><div class="bar" style="width:{max(1,t["us"]/mx*100):.0f}%"></div></td></tr>'
                for t in l["top"])
            findings["Latency"] = body + '</table></div>'
        except Exception as e:
            findings["Latency"] = f'<div class="card muted">unavailable: {core._esc(e)}</div>'
    html = core.export_report(lm, notes, findings)
    name = (lm.source_name or lm.model_id).replace("/", "_")
    return Response(content=html, media_type="text/html", headers={
        "Content-Disposition": f'attachment; filename="aifmri-{name}.html"'})


@app.get("/api/version")
def version():
    """Which build is actually being served (the UI header shows this too)."""
    return {"version": __version__}
