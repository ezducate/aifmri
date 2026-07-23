# AIFmri — functional imaging for neural networks

Drop a compiled model (.onnx, .pt/.pth, .h5/.keras/SavedModel) anywhere on
the page, see every layer in 3D, feed it a stimulus — an image, a sentence,
noise — and watch which parts of the network light up, fMRI-style. Click any
layer to read its raw floats, spatial activation maps, and to translate
latents into images or tokens.

## Run

    pip install -r requirements.txt
    uvicorn app.main:app --reload      # open http://127.0.0.1:8000

Five built-in samples (pure-ONNX, zero downloads) cover every modality and
multimodality. Beyond those, a **HuggingFace gallery** loads real
architectures — BERT-tiny, DistilBERT-SST2, ViT, Whisper-tiny, GPT-2 — with
one click: each is exported to ONNX on first use (cached after), the
tokenizer is auto-attached, and the default stimulus / normalization are
pre-filled. This is where the attention view really sings: ViT gives you a
197×197 patch-attention grid, BERT real wordpiece attention. Requires
`transformers` + `optimum` + `optimum-onnx` on the server.

## The fMRI toolkit

* **Image normalization presets** — unit (0…1), ImageNet, **CLIP / OpenAI**,
  Inception, and signed (−1…1), so real vision checkpoints get the exact
  preprocessing they expect.
* **Any modality** — the model's input signature is auto-classified as
  text / image / audio / video / tensor (dtype + rank + shape + name
  heuristics), and the stimulus panel switches to match:
  - **Image**: file upload, drag-and-drop, or a one-click webcam snapshot.
  - **Text**: tokenized via HF `tokenizer.json` or the built-in byte
    tokenizer; masks and token types auto-generated.
  - **Audio**: wav/flac/ogg/mp3 upload or a 4-second mic recording (encoded
    to WAV in the browser — no server-side ffmpeg needed for the mic).
    Fitted automatically to what the model wants: raw waveform (fixed or
    dynamic length) or log-mel spectrogram with mel bins / frame count read
    from the input shape. Whisper-style (1, 80, 3000) is recognized.
  - **Video**: any clip; frames sampled evenly to the model's time axis,
    fitted to NCTHW / NTCHW / NTHWC layouts.
  - Plus Gaussian noise and zeros baselines for any input.
* **Multimodal models** (CLIP-style, image × text, etc.): inputs are split
  into *stimulus inputs* — each gets its own stimulus card with its own
  mode, file, or text — and *companions* (attention masks, token types,
  position ids), which are auto-filled and follow the dynamic shape of the
  stimulus input they belong to. Pair this with contrast mode: hold a run
  as baseline, change only the caption, and see exactly which layers carry
  the text signal.
* **Contrast mode** — hold any run as a baseline, run a second stimulus, and
  view A − B on a diverging coolwarm map. This is how real fMRI works:
  activation is always relative to a baseline condition.
* **Sweep** — animate the activation wave through the network front-to-back.
* **Inspector** — per-layer stats (mean/std/sparsity/RMS), per-channel
  spatial heatmaps (token × dim heatmaps for transformers — attention
  matrices render directly), paginated raw float readout, render-any-tensor-
  as-image, and logits → top-k labels/tokens.

## Neuron attribution — what makes a unit fire? (v0.10)

Seeing a unit light up is only half the story; attribution answers *why*.
Select a layer (and optionally a specific channel via the channel selector),
then:

* **Occlusion map** — AIFmri perturbs regions of the actual input and measures
  how much the unit's response drops, producing a saliency map aligned to the
  stimulus: a patch grid over an image, a per-token highlight over text, or a
  sliding-window strip over audio. Bright = important to that unit. (Validated:
  on a test image with a bright square in one quadrant, the occlusion peak
  lands exactly on that quadrant.)
* **Rank stimuli** — ranks candidate inputs by how hard they fire the unit. If
  you've recorded a temporal sequence, it ranks those frames (click one to jump
  straight to it); otherwise it sweeps a batch of random inputs to surface the
  unit's empirical preference.

Together with the v0.9 stimulus viewer, this closes the loop: you can see the
input, see the activation, read what the layer is disposed toward, and now
trace a unit's firing back to the input regions that cause it.

Endpoints: `GET /api/models/{id}/attribution/occlusion?node=…&channel=…`,
`/attribution/rank_frames`, `/attribution/rank_noise`.

## Activation maximization — ask the neuron what it wants (v0.15)

Occlusion asks *"which part of this input matters?"*. Ranking asks *"which of my
inputs fires it hardest?"*. Maximization asks the unit directly: it **searches
input space for the stimulus that drives the unit as hard as possible**, and
synthesizes it.

ONNX graphs aren't autodiff-friendly, so instead of gradient ascent this uses an
**NES-style evolutionary search** — sample a population of perturbations, score
each with a forward pass, step along the fitness-weighted direction. A few
hundred forward passes, no gradients, works on any ONNX model.

* **Regularized** — images get blur + jitter during the search, and the input is
  norm-capped. This matters: without regularization the optimizer finds
  adversarial high-frequency noise that maximizes the unit while showing no
  structure. The norm cap is doing real work too — for a ReLU net activation
  scales linearly with input scale, so uncapped the search would just crank the
  input to infinity. Capped, it answers the meaningful question: *the most
  exciting input at a fixed energy budget*.
* **Discrete inputs** (token ids) get hill-climbing over vocabulary instead.
* **The result becomes the live stimulus** — the server runs the synthesized
  input, so the 3D view, the inspector, and the stimulus viewer all repaint to
  show the network responding to what the neuron wanted. Click Maximize and the
  brain is now looking at the neuron's dream.
* An **optimization trace** plots the response climbing over the search.

*Validation:* on demo-cnn, `relu2[3]` climbs **0.067 → 1.098 (16.4×)** in ~1.4s
— 16× stronger than a real noise image — and different channels synthesize
genuinely different images (mean pixel difference 17–44), which is the point:
each unit wants its own thing.

Endpoint: `POST /api/models/{id}/attribution/maximize` (node, channel, steps,
pop, sigma, lr, regularize, seed).

## Network health scan — is the model wasting capacity? (v0.16)

Every other view in AIFmri is single-stimulus. The health scan probes the model
with a **batch of varied stimuli** and aggregates per-unit statistics across
them, which is the only way to see the pathologies that actually bite:

* **dead** — the unit never fires for *any* stimulus (the classic dead ReLU)
* **weak** — it fires, but negligibly next to its layer-mates
* **constant** — it never varies across stimuli, so it carries no information
* **duplicate** — two units whose activation *patterns* are near-perfectly
  correlated: the layer is narrower than it looks

Results come back as a verdict, a dead-percentage profile by depth, and a
ranked list of problem layers you can click to jump straight to.

Two details that make it trustworthy rather than decorative:

* **Duplicates compare patterns, not magnitudes.** The first implementation
  correlated each channel's mean activation and reported 54 duplicate pairs in a
  layer that had exactly 1 — because channels with similar average energy look
  alike by that measure. Correlating each unit's actual activation *pattern*
  across the probe batch fixed it.
* **Structural ops are skipped.** Flatten/Reshape/Transpose have no units of
  their own — they're views of the previous layer, so "dead units" there is a
  meaningless restatement. Excluding them removed the last false positives.

*Validation:* against a model with deliberately sabotaged weights (three conv
channels forced permanently dead via bias, and two filters made identical), the
scan reports **exactly 3 dead channels** and **exactly 1 duplicate pair —
correctly identified as the planted pair, correlation 1.0** — while the healthy
model reports **0 dead and 0 duplicates**.

Endpoint: `GET /api/models/{id}/health?n=…&dup_thresh=…`.

## Weights — what the network knows (v0.17)

Every other view is about activations: the network *responding*. Weights are
the other half — the learned structure itself. Select a layer and the inspector
shows its parameter tensors, distribution, ‖w‖, zero-fraction, and — for
convolutions — a **kernel contact sheet**. First-layer RGB kernels render in
colour (the classic view); deeper kernels have many input channels and no
honest colour mapping, so each filter is shown as its mean-over-input-channels
map on the inferno scale. With a channel selected you also get that filter's
norm and its rank among its layer-mates.

*Cross-validation:* for a first-layer conv the maximizer and the filter should
agree — at fixed input norm the input that maximizes a linear filter **is** the
filter (matched-filter theory). They do: synthesized inputs correlate with
their own filter about 2x more strongly than with a different one. Two
independent features, one answer.

Endpoints: `GET /api/models/{id}/weights?node=…`, `/weights/image?node=…`.

## Latency lens (v0.17)

The same 3-D view, painted by **time** instead of activation: not "what lit
up?" but "where does the time actually go?". Timings are real, from ONNX
Runtime's own profiler over repeated runs.

Two honesty details worth knowing, both surfaced in the UI rather than hidden:

* **ORT fuses ops.** Conv+Relu becomes one kernel named after the *last* node,
  so that Relu's time really includes the Conv before it. Fused layers are
  flagged with the op ORT actually ran.
* **ORT does work your graph doesn't contain** — layout reorders, memcpy. That
  time is reported as overhead rather than silently dropped.

Endpoint: `GET /api/models/{id}/latency?runs=…`.

## Session export (v0.17)

Findings used to die with the browser tab. **Download report** snapshots the
session — model, stimulus (image or token chips), full activation profile, plus
optional health and latency sections — into a **single self-contained HTML
file** with images inlined as data URIs. No server, no assets, no dependencies:
open it anywhere, e-mail it, attach it to a PR.

Endpoint: `POST /api/models/{id}/export` (notes, include_health, include_latency).

## Causal language models (v0.17)

Real generative LMs (DistilGPT-2) now load and run. The blocker was never the
KV-cache I'd guessed at — it was that causal-LM exports emit **legal zero-size
intermediate tensors**, and every reduction over them (`min`, `mean`) threw.
Guarding zero-size activations fixed it in one line, and DistilGPT-2 now works
throughout: real BPE tokens, **6 attention layers with genuine causal masking**,
token-by-token temporal recording, health, and latency.

The J-lens and circuit tracer are the exception, and they say so: they rebuild a
forward subgraph in memory (~3x the model size), so on a 482 MB model they'd
exhaust RAM. Rather than let the OS OOM-kill the server they refuse cleanly and
explain why, and they still work on anything within available RAM. The gallery's GPT
entry is now real DistilGPT-2 rather than an untrained stub — every gallery
model is now genuinely trained.

## Weights, latency, export, and causal LMs (v0.17)

**Weight / filter viewer.** The tool showed what the network *does*; this shows
what it *knows*. Per layer: kernel contact sheets (RGB tiles for a 3-channel
first layer, inferno mean-maps deeper), weight histograms, and per-filter norms
ranked so a dead or degenerate filter stands out.

*Cross-validated against the maximizer, which is the satisfying part.*
Matched-filter theory says the input that maximally excites a first-layer conv
filter should be that filter's own pattern. The maximizer is an evolutionary
search that never sees the weights; the weight viewer reads them directly.
Measured sign-invariantly (per-patch |correlation|, because the mean|act|
objective lets the optimum flip sign position-to-position), **all 5 tested
channels' synthesized inputs match their own filter**, 1.5-1.6x above any other
filter. Two independent features agreeing via theory.

**Latency lens.** Real ONNX Runtime per-node profiling — paint the 3D view by
time instead of activation and the bottleneck is obvious. It is honest about
what it can't see: ORT **fuses** ops (on demo-cnn it folds Relu into Conv, and
the report says so rather than pretending "Relu = 41% of runtime" is a fact),
and ORT-internal ops are reported separately as overhead rather than blamed on
your layers.

**Session export.** A self-contained HTML report — stimulus, activation
profile, optional health and latency sections, your notes — with every image
base64-inlined and zero external references. It opens offline, forever.

**Causal LMs now work — and my earlier diagnosis was wrong.** I'd said GPT-2
failed because of "KV-cache size-0 inputs". The model was never the problem:
causal-LM exports legitimately produce *empty* intermediate tensors, and
AIFmri was calling numpy reductions (`.min()`) on them, which throws. A
one-line guard skipping zero-size arrays fixed it. DistilGPT-2 (1366 layers)
now loads, runs, records, scans and profiles — and its attention comes out
**perfectly lower-triangular** (strictly-upper mass 0.000000), which is the
causal mask emerging from the data with nothing in the code assuming it.

*The honest limit:* the J-lens and circuit tracer rebuild a forward subgraph in
memory. Measured cost is **~5x the model size** (4.9x on BERT-tiny, 5.2x on
ViT-tiny) — not the 3.5x I first assumed, which OOM-killed the process when I
tried it. The guard is now RAM-aware rather than a fixed cap: it measures free
memory and refuses only when the rebuild genuinely won't fit, so a 482 MB
DistilGPT-2 lens is out on a 4 GB box but fine on a workstation. Everything
else works at any size.

## Circuit tracing — causal influence between layers (v0.11)

Attribution traces a unit back to the input; circuit tracing traces influence
*forward*, between layers, measuring genuine causal effects rather than
correlations. Select a source layer (and optionally a channel), then:

* **Boost → trace** — scales the source unit up and runs forward to a chosen
  deeper target layer, reporting the target channels that move most (with
  signed deltas). This is the causal downstream "wire": what this unit
  actually drives.
* **Ablate → output** — zeroes the source unit and measures the impact on the
  model's final output: the L2 change in logits, which classes/tokens shift
  most, and crucially **whether the prediction flips**. It's the direct test
  of how much the model relies on that unit.

Implemented by reusing the J-lens subgraph runner: the source layer becomes a
graph input, its activation is perturbed, and the model runs forward from
there to the target (any external dependencies are fed from the recorded
activations). Validated: boosting a source channel produces ranked downstream
movement; ablating a whole layer produces a measurable, correctly-signed logit
shift. A guard refuses targets that aren't deeper than the source.

Endpoints: `GET /api/models/{id}/circuit/targets?source=…`,
`/circuit/trace?source=…&target=…`, `/circuit/ablate?source=…`.

## Model diffing — what did the fine-tune change? (v0.12)

Load two related models (a base and a fine-tuned / edited variant), run the
**same stimulus** through both, and see exactly where they diverge. AIFmri
replays model A's exact input tensors through model B, matches layers (by node
id for shared architectures, falling back to depth+op for renamed graphs), and
computes per-layer divergence — relative L2 change and cosine similarity of the
two activation vectors — plus whether the final prediction changed.

* **Per-depth divergence bars** — a compact profile, shallow to deep, coloured
  by how much each layer's activations moved. A fine-tune that only touched the
  classifier head shows a flat-zero profile with a single spike at the end; a
  full fine-tune shows divergence rippling from the first changed layer forward.
* **Most-changed layers** — ranked, clickable to jump straight to the layer in
  3D.
* **Output comparison** — the two models' predictions side by side, flagged when
  they differ.

*Validation:* perturbing only a model's classifier-head weights produces a diff
that reads exactly `0.000` through every conv/pool/relu layer and `1.745` at the
final Gemm — the change is localized precisely to where it was made.

Endpoints: `GET /api/diff?model_a=…&model_b=…`, `GET /api/models` (lists loaded
models available to diff against).

## Wiring view (v0.13)

Edges aren't drawn as a single line between layer centres any more — each edge
is a **bundle of unit-to-unit lines whose pattern reflects the operator**:

* **Gemm / MatMul** → all-to-all fan. A fully connected layer looks fully
  connected: every sampled unit on the left wires to every sampled unit on the
  right.
* **Conv / Pool** → local receptive-field fans (each target unit draws from a
  small neighbourhood of the source), so you can see locality at a glance.
* **Elementwise** (Relu, Add, …) → parallel 1:1 lines.

**Pyramids** mode goes volumetric instead of linear. A fully-connected layer is
drawn as one open, square-based pyramid per sampled *source* unit — apex on the
unit, base opening across the whole target slab, i.e. "this unit reaches
everything". The base is *square* because the thing it opens onto is a square
slab of units; a round cone would misrepresent the footprint it actually
covers, and the geometry is rolled 45° so the base's edges line up with the
slab rather than sitting on it as a diamond. Rendered with additive blending,
overlapping pyramids build up density where connectivity is dense, conveying an
all-to-all Gemm with one primitive per unit instead of k² lines. Conv/pool get
the mirror image: a narrow pyramid per *target* unit opening back onto its
receptive field, so locality reads as a tight beam and density as a wide glow.
The fans are tinted by their source layer's activation, so they light up with
the network.

Four modes in the **WIRING** group of the view bar: **Simple** (one line per
edge), **Bundled** (default), **Dense**, **Pyramids**. The wires are *sampled, not
exhaustive* — an honest approximation, budgeted per scene (bundled ≈7k
segments, dense ≈30k) with a per-edge cap so one fat Gemm can't eat the whole
budget. On a 205-node BERT that lands at ~6.7k / ~16k segments with layout
morphs still smooth. The bundles follow layers through layout transitions, so
helix and radial stay wired correctly.

## Creative fixes for the hard cases

* **Bare state_dicts** (weights with no architecture): the server
  fingerprints the key names and tensor shapes against the torchvision zoo
  and offers ranked candidates — a resnet18 state_dict is identified with a
  122/122 shape match in ~5 s. One click rebuilds, loads the weights, and
  converts. If it's a custom model, paste the class code in the recovery
  panel instead (runs locally on your own machine, on your own file).
* **Dynamic / symbolic shapes**: symbolic dims are resolved by name
  (`batch`→1, `seq*`→32, image-like→224), validated by a zero-stimulus probe
  run at load time, and shown in an editable Inputs table — fix any dim and
  re-probe without reloading. Dynamic sequence axes stay dynamic: text
  stimuli run at their natural token length.
* **Multi-input models**: the primary input is auto-detected (ids over
  masks); companion inputs are filled by name convention.

## v0.20 — the bugs a real browser found

A browser agent drove the actual UI for ten minutes and found six things that
114 automated tests had all passed. Worth recording *why* they were missed:

* **Whole control groups were unclickable.** The floating bars were centred on
  the *window*, so at <=1500px — i.e. a laptop — the LAYOUT group and the
  carpet's play button slid under the 312px left rail. The rails sit at z-12
  (raised in v0.9 so the analytics bar would stop eating the inspector's
  buttons), so the rail won and swallowed the clicks. Two of my own fixes
  colliding. The bars now centre on the **gap between the rails**, and the HUD
  slides left when the inspector opens.
  *Why the tests missed it:* they clicked with `force=True`, which tells
  Playwright to skip its actionability check — a real mouse cannot force-click
  — and they ran at 1600px+, where the bug does not occur.
* **Stale readouts after switching models.** `updateHUD()` early-returns when
  there is no activation, so the previous network's counters just stayed on
  screen describing a model that was no longer loaded, with nothing marking
  them stale. `clearScene()` now wipes the counters, plots, carpet and timeline.
* **Attention arcs never rendered on a real sentence.** The cut was absolute
  (`w > 0.08`), but attention rows are a softmax summing to 1: at 22 tokens the
  mean weight is 0.045, so *zero* arcs were drawn. It only worked under ~12
  tokens — which is exactly the length every test and demo used. Now each query
  contributes its top-k keys, with a floor relative to that row's own max.
* **Camera zoom could lock up permanently.** OrbitControls had no distance
  bounds, so you could dolly onto the target, leaving its multiplicative zoom
  nothing to scale; zooming back out then did nothing forever. Bounded to
  6..6000. And the fly-to lerp never actually arrived — it stopped at ~94% when
  its timer expired, which is why "Overview" appeared to fix the angle but not
  the zoom. It now lands exactly.

`pytest -m ui` is the lesson made permanent: it drives a real browser at
1280/1366/1500/1920, never uses `force`, and asserts reachability with
`document.elementFromPoint` — "is this control the element the mouse would
actually hit?"

## Tests

```bash
pip install pytest
pytest                # 114 tests, ~7s
pytest -m ui          # + real-browser reachability (needs playwright + chromium)
pytest -m slow        # + real HuggingFace models (downloads, needs RAM)
```

These aren't smoke tests — they encode the invariants that actually caught
bugs, so a future edit can't quietly break them:

* **`test_health.py`** — scans a model with *planted* pathologies (three conv
  channels forced dead by bias, two filters made byte-identical) and demands
  exactly 3 dead and exactly 1 duplicate pair, correctly named. Pins the two
  bugs that shipped once: duplicates computed from magnitude instead of
  pattern (54 false pairs where 1 existed), and Flatten inventing phantom dead
  units.
* **`test_attribution.py`** — the matched-filter cross-validation: the
  maximizer never sees the weights, yet must rediscover each conv filter. Also
  pins *how* to measure it — the sign-invariant per-patch score, because the
  naive average-patch score reports a false failure.
* **`test_diff.py`** — perturbing only the classifier head must read 0.000
  divergence upstream and spike at the Gemm.
* **`test_robustness.py`** — zero-size intermediates (the causal-LM bug,
  reproduced synthetically in milliseconds instead of a 480 MB download),
  non-finite activation sanitizing, and LRU model eviction.
* **`test_attention.py`** — detection precision (demo-clip's similarity MatMul
  must NOT be flagged), softmax rows summing to 1, fused-attention fallback.
* **`test_frontend.py`** — `node --check` on the module script (a bad edit once
  deleted a function declaration and the app died silently on load), plus every
  `$('id')` the JS touches must exist in the DOM.
* **`test_gallery_slow.py`** — DistilGPT-2's attention must come out strictly
  lower-triangular. Nothing in the code assumes causality, so that can only
  come from the model.

## Architecture

    app/core.py      ingestion, shape resolution, tokenizers, fingerprinting
    app/samples.py   built-in demo models (pure ONNX, generated on demand)
    app/main.py      FastAPI endpoints
    app/static/      three.js viewer — one InstancedMesh per layer

Everything converges on ONNX; activation capture promotes every intermediate
value to a graph output so one onnxruntime call records the whole brain.
torch/torchvision/tensorflow are lazy imports — only needed for those inputs.

## API

    POST /api/models                    upload (returns needs_arch for state_dicts)
    POST /api/models/resolve            rebuild from arch name or pasted code
    GET  /api/samples · POST /api/samples/{name}
    POST /api/models/{id}/shapes        edit input shapes, re-probe
    POST /api/models/{id}/run           noise | image | text | audio | video | zeros
    POST /api/models/{id}/run_multi     per-input stimuli (multimodal models)
    POST /api/models/{id}/tokenizer     byte | hf (tokenizer.json)
    GET  /api/models/{id}/raw           raw float slices
    GET  /api/models/{id}/spatial       heatmaps (channel or token × dim)
    GET  /api/models/{id}/decode/image  render latent as PNG
    GET  /api/models/{id}/decode/topk   logits → labels / tokens
    POST /api/models/{id}/labels        attach class labels
