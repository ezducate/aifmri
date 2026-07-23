"""The frontend is one big module script inside index.html.

A syntax error there is invisible to the Python tests and fatal in the browser
— a bad string-replace once silently deleted `function buildScene(){` and the
whole app died on load. `node --check` catches that class of bug instantly.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

INDEX = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"


def _module_source() -> str:
    html = INDEX.read_text()
    m = re.search(r'<script type="module">(.*?)</script>\s*</body>', html, re.S)
    assert m, "module script not found in index.html"
    return m.group(1)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_frontend_module_parses(tmp_path):
    f = tmp_path / "mod.mjs"
    f.write_text(_module_source())
    r = subprocess.run(["node", "--check", str(f)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_every_wired_element_id_exists_in_the_dom():
    """Catches handlers bound to an id that was renamed or never added."""
    html = INDEX.read_text()
    src = _module_source()
    declared = set(re.findall(r'id="([A-Za-z0-9_]+)"', html))
    used = set(re.findall(r"\$\('([A-Za-z0-9_]+)'\)", src))
    missing = used - declared
    assert not missing, f"JS references ids that do not exist: {sorted(missing)}"


def test_no_browser_storage_apis():
    """Artifacts/sandboxes disallow them and they fail silently."""
    src = _module_source()
    assert "localStorage" not in src and "sessionStorage" not in src


def test_wiring_modes_are_all_reachable():
    html = INDEX.read_text()
    for mode in ("simple", "bundled", "dense", "pyramids"):
        assert f'data-wire="{mode}"' in html


def test_the_fan_is_a_square_based_pyramid():
    """The base opens onto a SQUARE slab of units, so it is a pyramid, not a
    cone — 4 radial segments, rolled 45 degrees to put the base's edges (not
    its vertices) on the axes so it lines up with the slab."""
    src = _module_source()
    assert "ConeGeometry(1,1,4,1,true)" in src, "4 sides = square base"
    assert "fanGeo.rotateY(Math.PI/4)" in src, "align the base with the slab"


def test_errors_are_toasted_not_just_written_below_the_fold():
    """The status box sits at the BOTTOM of a long scrollable rail. An error
    raised by a control at the TOP (the gallery) rendered off-screen and the
    click looked like it did nothing."""
    src = _module_source()
    assert "function toast(" in src
    assert "if(cls==='err') toast(m,'err');" in src, "status() must toast errors"
    assert 'id="toast"' in INDEX.read_text()


def test_gallery_declares_missing_exporter_up_front():
    src = _module_source()
    assert "available" in src and "hint" in src
    assert "disabled" in src, "gallery buttons must disable when unusable"
