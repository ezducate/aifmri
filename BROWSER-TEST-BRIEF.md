# AIFmri — browser test brief

**Paste this whole file into Claude in Chrome, with the app open at
`http://127.0.0.1:8001`.**

---

## Your job

You are testing a local web app called **AIFmri** — a tool that loads neural
network files and visualises them in 3D, showing which layers "light up" for a
given input, like an fMRI. It is a research tool, single-user, running on
localhost. Nothing you do can hurt anyone.

I wrote it and I have tested the *backend* heavily (114 automated tests). What I
have **not** been able to test properly is the **real browser**: actual WebGL,
actual scrolling, actual click targets, actual window sizes. That is what you
are for. My automated tests drive a headless browser and have repeatedly missed
things a human would hit in ten seconds.

**Work through the scenarios below in order. For each one, report PASS or FAIL.
Be sceptical and specific. I would much rather hear "this looked broken" than
have you be polite about it.**

---

## Before you start

1. **Check the version badge.** Next to the `AIFMRI` title, top-left, there
   should be a small grey `v0.19.3`.
   * If it says anything else, or nothing: **stop and tell me.** The wrong
     build is being served and every other result would be meaningless.
2. **Open the browser devtools console** (F12) and leave it open. Report *any*
   red error at any point, along with what you had just clicked.
3. **Do not install anything, do not edit files, do not run terminal commands.**
   You are only clicking and looking.
4. The left rail **scrolls**, and it is long. Several panels are collapsed
   `<details>` sections you must click open. If you cannot find a control, scroll
   the left rail before concluding it is missing.

---

## Known limitations — DO NOT report these as bugs

These are deliberate. If you report them, the real findings get buried.

| You will see | Why it is intended |
|---|---|
| The **HuggingFace gallery** buttons are greyed out with a red banner saying `pip install transformers optimum …` | The export library is not installed on this machine. The banner is the *correct* behaviour. Do not try to install it. |
| **"Read this layer"** (J-lens) says the output is *"a feature/hidden vector, not a vocabulary or labelled class head"* | Correct refusal. The J-lens needs class labels — see Scenario 7 for how to give it some. |
| Demo models predict nonsense (`digit_7` for a blank image) | The demo samples have **random untrained weights**. Structure is real; predictions are meaningless. Not a bug. |
| A model refuses the J-lens saying `This model is NNN MB … only NNN MB is free` | Deliberate memory guard. |
| Text stimulus tokens look like `t h e   c a t` | Correct — the demo transformer is byte-level. Its vocabulary *is* the 256 byte values. |

---

## Scenario 1 — First load

1. Reload the page.
2. **Expected:** left rail shows a list of 5 demo models (`Demo CNN`,
   `Demo transformer`, `Demo audio net`, `Demo video net`, `Demo CLIP`), and
   below them a HuggingFace gallery section.
3. The 3D canvas should be dark with a faint grid. No layers yet.

**Report:** does anything overlap, clip, or sit off-screen? Is any text
unreadable or cut off?

---

## Scenario 2 — The core loop (most important scenario)

1. Click **Demo CNN**.
2. **Expected:** 8 dark slabs appear in the 3D view. A **view bar** appears
   top-centre. Four counters appear top-right. Plots appear along the bottom.
3. In **STIMULUS**, set Source to **Gaussian noise**.
4. Click the big yellow **Run stimulus**.
5. **Expected:** the slabs light up orange/yellow. The top-right counters fill
   in (`ACTIVE LAYERS`, `SPARSITY`, `PEAK |ACT|`). The bottom `LAYER STRIP` and
   `ACTIVITY BY DEPTH` plots fill in.
6. Click any slab **in the 3D view**.
7. **Expected:** the right-hand inspector opens with the layer's name, stats,
   a histogram, a channel grid, and a spatial map.

**Report:** did clicking a slab in 3D actually select it? Does the inspector
match the layer you clicked?

---

## Scenario 3 — Camera and 3D interaction

1. Drag on the canvas to orbit. Scroll to zoom. Right-drag to pan.
2. Click **Overview**, **Top**, **Side** in the CAMERA group.
3. Click **Helix**, then **Radial**, then **Layered** in the LAYOUT group.

**Report:** is the orbit smooth or jerky? Do the layout changes animate or jump?
Does anything fly off-screen or disappear? Does `Overview` actually re-frame the
network?

---

## Scenario 4 — Wiring (a recent change, please be harsh)

1. With Demo CNN loaded and a stimulus run, click **Side** in CAMERA.
2. In the **WIRING** group, click through: **Simple** → **Bundled** → **Dense**
   → **Pyramids**.
3. After each, read the status message.

**Expected:** each mode visibly changes the lines between layers. `Simple` is one
thin line per layer pair. `Bundled` is many. `Dense` is more. `Pyramids` replaces
lines with translucent glowing volumes fanning from each unit onto the next layer.

**Report specifically:**
* Does **Pyramids** actually look like square-based pyramids opening onto the
  layer, or does it look like a mess / a solid wall / nothing at all?
* Do the pyramids **glow brighter** where the network is more active?
* Does switching modes ever freeze the page? For how long?
* **Judge it aesthetically.** Does it help you understand the network, or is it
  just noise? I cannot see the screen and this is exactly the kind of thing I
  cannot verify. Say so bluntly if it looks bad.

---

## Scenario 5 — Attention (transformer)

1. Click **Demo transformer**.
2. Make sure the **Tokenizer** dropdown says *"Byte-level (built-in, any model)"*.
3. Type a sentence in **Text stimulus**, click **Run stimulus**.
4. In the **LAYERS** list, type `softmax` in the filter box, click the result.

**Expected:** the inspector shows an **Attention** section with a heatmap, a
head selector, and a token list. In 3D, glowing arcs appear over a ring of
token dots.

**Report:** are the arcs visible and comprehensible? Is the heatmap square?
Do the tokens under it match your sentence?

---

## Scenario 6 — Temporal recording

1. Still on Demo transformer. Scroll to **TEMPORAL · RECORD A SEQUENCE**.
2. Sequence source should say *"Sentence → token by token"*.
3. Type a longer sentence in the box below it. Frames: `24`. Click **⏺ Record**.
4. **Expected:** a **BOLD CARPET** panel appears at the bottom — a colourful
   grid, layers × time — with a play button and a slider.
5. Press **▶**. Then drag the slider. Then click directly on the carpet.

**Report:** does playback animate the 3D view? Does scrubbing feel instant or
laggy? Does clicking the carpet jump to that frame *and* select that layer?
Does the attention heatmap grow as more tokens are revealed?

---

## Scenario 7 — The analysis tools (these are buried; find them)

1. Click **Demo CNN**, run a **Gaussian noise** stimulus.
2. Open the **Decode extras** section in the left rail (it is collapsed).
3. Click **Choose File** and pick **`samples/digit_labels.json`** — it ships
   inside the app folder, next to `START.bat`. Then click **Attach labels**.
   *(Without labels the J-lens correctly refuses with a "feature vector"
   message — that is expected, not a bug.)*
4. Select layer `relu2` from the LAYERS list.
5. Scroll the **right-hand inspector** all the way down. You should find, in order:
   * **Stimulus that produced this** — should show the input image
   * **Jacobian lens** — buttons *Read this layer*, *Scan all layers*
   * **Neuron attribution** — *Occlusion map*, *Rank stimuli*, *✦ Maximize*
   * **Circuit tracing** — a target dropdown, *Boost → trace*, *Ablate → output*
6. Click **every one of those buttons** and report what happens.

**Report — this is the scenario I most want data on:**
* Can you actually **reach and click** each button, or does something overlap
  them? (The bottom analytics bar has covered these before.)
* Does each one produce visible output, or does it look like nothing happened?
* Time them. Anything over ~3s with no feedback is a bug — tell me which.
* **✦ Maximize** should repaint the 3D view and the stimulus image. Does it?

---

## Scenario 8 — Health, weights, latency, export

1. In the left rail, open each of these collapsed sections and use them:
   * **Network health** → set Probes `24` → **Scan**
   * **Latency lens** → **Profile**
   * **Model diff** → (needs 2 models; see Scenario 9)
   * **Export session** → type a note → **Download report**
2. Also: with a layer selected, the inspector should show **weights** — a
   kernel image for `conv1`.

**Report:** does the health scan give a verdict and clickable problem layers?
Does Profile show per-layer times? Does the export actually **download an HTML
file**, and does opening that file show a readable report with images?

---

## Scenario 9 — Two models at once

1. Click **Demo CNN**. Run a stimulus.
2. Click **Demo CNN again** (loads a second copy).
3. Open **Model diff**, pick the other model in the dropdown, click **Diff A → B**.

**Expected:** it reports layers matched and near-zero divergence (they are
identical models).

**Report:** does the dropdown actually list the other model? Any 404?

---

## Scenario 10 — Try to break it

Please actually try. Ten minutes of malice is worth more than my whole test suite.

* Resize the window very narrow (~800px). Then very wide. What breaks?
* Click **Run stimulus** repeatedly, fast.
* Click **Record** and immediately click other things while it runs.
* Load a different model *while* something is recording.
* Switch to **Pyramids** on the transformer (15 layers) and then to **Dense**.
* Click a layer, then load a new model — does the stale inspector clear?
* Leave a recording playing and switch layouts.
* Scroll both rails to the bottom and try to click things near the edges.
* Zoom the browser to 150% and to 67%. Does the layout survive?

---

## How to report back

For each scenario: **PASS** or **FAIL**, one line.

For anything that failed or looked wrong, give me:

1. **What you did** — exact clicks
2. **What you expected**
3. **What actually happened**
4. **Console errors** — the red text, verbatim
5. **Severity** — *broken* (feature unusable) / *ugly* (works, looks bad) /
   *confusing* (works, unclear)
6. **A screenshot** if it is visual

At the end, tell me the **three worst problems** in priority order, and — since
you can see the screen and I cannot — your honest opinion of whether this thing
is actually pleasant to use.
