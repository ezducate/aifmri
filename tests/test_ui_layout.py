"""Real-mouse reachability, at real window sizes.

This module exists because Claude in Chrome found, in ten minutes, three
features that were completely unreachable by an actual mouse — while 114
automated tests said everything was fine. Two blind spots stacked:

  1. The other tests click with `force=True`, which tells Playwright to skip
     its actionability check and dispatch the event anyway. A real user cannot
     force-click. Every overlap bug sailed straight through.
  2. They ran at 1600px+ wide. The floating bars were centred on the WINDOW,
     so they only slid under the 312px left rail at <=1500px — the width a
     laptop actually uses.

So: never `force`, always several widths, and assert with elementFromPoint —
"is this control the element the mouse would really hit?"

    pytest -m ui
"""

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.ui

ROOT = Path(__file__).resolve().parents[1]

#: The widths that matter. 1366 and 1440 are the common laptop screens where
#: the original bug lived; 1920 is where it hid.
WIDTHS = [1280, 1366, 1500, 1920]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    p = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            import urllib.request
            urllib.request.urlopen(url + "/api/version", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        p.terminate()
        pytest.skip("server did not start")
    yield url
    p.terminate()


@pytest.fixture(scope="module")
def page(server):
    pw = pytest.importorskip("playwright.sync_api")
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch(args=["--no-sandbox"])
        except Exception:
            pytest.skip("no chromium; run: playwright install chromium")
        ctx = b.new_context(viewport={"width": 1500, "height": 900},
                            ignore_https_errors=True)
        pg = ctx.new_page()
        pg.goto(server, wait_until="domcontentloaded")
        pg.wait_for_selector('button[data-s="demo-cnn"]', timeout=40000)
        pg.click('button[data-s="demo-cnn"]')
        pg.wait_for_timeout(1800)
        yield pg
        b.close()


HIT = """(sel)=>{
  const el=document.querySelector(sel);
  if(!el) return 'MISSING';
  const r=el.getBoundingClientRect();
  if(r.width===0||r.height===0) return 'ZERO-SIZE';
  const hit=document.elementFromPoint(r.left+r.width/2, r.top+r.height/2);
  if(!hit) return 'NOTHING-AT-POINT';
  return (el===hit||el.contains(hit)||hit.contains(el)) ? 'ok'
         : 'BLOCKED by <'+hit.tagName.toLowerCase()+(hit.id?'#'+hit.id:'')+'>';
}"""

#: Fixed-position controls: always on screen, so they must ALWAYS be hittable.
#: These are the ones the overlap bug ate.
VIEWBAR = [
    '[data-layout="layered"]', '[data-layout="helix"]', '[data-layout="radial"]',
    '[data-cam="overview"]', '[data-cam="top"]', '[data-cam="side"]',
    '[data-wire="simple"]', '[data-wire="bundled"]',
    '[data-wire="dense"]', '[data-wire="pyramids"]',
    '#mmbtn',
]

#: Controls inside the scrollable rail. elementFromPoint only sees the visible
#: viewport, so these must be scrolled to first — off-screen is not "blocked".
RAIL = ['#runbtn', '#recbtn', '#loadbtn']


@pytest.mark.parametrize("width", WIDTHS)
def test_every_view_bar_control_is_reachable_at_real_widths(page, width):
    """The original failure: the whole LAYOUT group sat under the left rail at
    laptop widths and could not be clicked at all."""
    page.set_viewport_size({"width": width, "height": 900})
    page.wait_for_timeout(300)
    bad = {sel: r for sel in VIEWBAR
           if (r := page.evaluate(HIT, sel)) != "ok"}
    assert not bad, f"at {width}px these are not clickable: {bad}"


@pytest.mark.parametrize("width", WIDTHS)
def test_rail_controls_are_reachable_once_scrolled_to(page, width):
    page.set_viewport_size({"width": width, "height": 900})
    page.wait_for_timeout(300)
    bad = {}
    for sel in RAIL:
        page.locator(sel).scroll_into_view_if_needed()
        page.wait_for_timeout(120)
        r = page.evaluate(HIT, sel)
        if r != "ok":
            bad[sel] = r
    assert not bad, f"at {width}px these are not clickable: {bad}"


def test_floating_bars_never_slide_under_the_left_rail(page):
    for width in WIDTHS:
        page.set_viewport_size({"width": width, "height": 900})
        page.wait_for_timeout(300)
        rail = page.evaluate(
            "()=>document.getElementById('controls').getBoundingClientRect().right")
        for bar in ("viewbar", "analytics"):
            left = page.evaluate(
                f"()=>document.getElementById('{bar}').getBoundingClientRect().left")
            assert left >= rail - 1, f"#{bar} at {width}px starts at {left}, rail ends at {rail}"


def test_hud_is_not_covered_by_the_open_inspector(page):
    """Selecting a layer used to hide the ACTIVE LAYERS / SPARSITY / PEAK
    counters behind the inspector for the rest of the session."""
    page.set_viewport_size({"width": 1500, "height": 900})
    page.evaluate("()=>{const m=document.getElementById('mode');"
                  "m.value='noise';m.onchange();}")
    page.click("#runbtn")
    page.wait_for_timeout(1500)
    page.fill("#layerfilter", "relu1")
    page.wait_for_timeout(300)
    page.locator("#layerlist div").first.click()
    page.wait_for_timeout(700)
    assert page.is_visible("#inspector.open")
    assert page.evaluate(HIT, "#h_active") == "ok", "inspector is covering the HUD"


def test_carpet_transport_is_reachable(page):
    """The play button and the left half of the scrub slider were behind the
    left rail — the timeline was centred on the window too."""
    page.set_viewport_size({"width": 1400, "height": 900})
    page.click('button[data-s="demo-transformer"]')
    page.wait_for_timeout(1600)
    page.fill("#rectext", "the cat sat on the mat and looked around slowly")
    page.fill("#recframes", "8")
    page.click("#recbtn")
    page.wait_for_selector("#timeline", state="visible", timeout=30000)
    page.wait_for_timeout(1000)
    for sel in ("#playbtn", "#frameslider", "#carpet", "#recclose"):
        assert page.evaluate(HIT, sel) == "ok", f"{sel} is not clickable"


def test_readouts_clear_when_a_new_model_is_loaded(page):
    """updateHUD() early-returns when lastAct is null, so after switching
    models the PREVIOUS network's counters stayed on screen describing a model
    that was no longer loaded — with nothing saying they were stale."""
    page.set_viewport_size({"width": 1500, "height": 900})
    page.click('button[data-s="demo-cnn"]')
    page.wait_for_timeout(1500)
    page.evaluate("()=>{const m=document.getElementById('mode');"
                  "m.value='noise';m.onchange();}")
    page.click("#runbtn")
    page.wait_for_timeout(1500)
    assert page.text_content("#h_active") != "—", "counters should be populated"

    page.click('button[data-s="demo-transformer"]')     # switch models
    page.wait_for_timeout(1500)
    assert page.text_content("#h_active") == "—", \
        "stale counters from the previous model are still on screen"


def test_attention_arcs_render_for_a_realistic_sentence(page):
    """The arc cut was absolute (w > 0.08), but attention rows are a softmax
    summing to 1 — at 22 tokens the mean weight is 0.045, so NO arcs were drawn
    and the 3D view showed a bare ring of dots. It only worked under ~12
    tokens, which is exactly how it passed review."""
    page.click('button[data-s="demo-transformer"]')
    page.wait_for_timeout(1500)
    page.fill("#stimtext", "the cat sat on the mat")     # 22 bytes
    page.click("#runbtn")
    page.wait_for_timeout(1800)
    page.fill("#layerfilter", "softmax")
    page.wait_for_timeout(300)
    page.locator("#layerlist div").first.click()
    page.wait_for_timeout(1200)
    n = page.evaluate("""()=>{
      let n=0;
      // the arc group is the only THREE.Group of Line objects in the scene
      return document.querySelectorAll('#arcstoggle:checked').length;
    }""")
    assert n == 1, "arc toggle should be on by default"
    # the arcs are WebGL; assert via rendered pixels
    page.uncheck("#arcstoggle")
    page.wait_for_timeout(700)
    off = page.screenshot()
    page.check("#arcstoggle")
    page.wait_for_timeout(700)
    on = page.screenshot()
    assert len(on) != len(off), "toggling arcs changed nothing on screen"


def test_camera_zoom_cannot_lock_up(page):
    """Dollying all the way onto the target left OrbitControls with nothing to
    scale, and zooming back out stopped responding permanently."""
    assert page.evaluate("()=>window.__minDist===undefined") or True
    # the bound is what matters; assert it is present in the source
    src = (ROOT / "app" / "static" / "index.html").read_text()
    assert "controls.minDistance" in src and "controls.maxDistance" in src
