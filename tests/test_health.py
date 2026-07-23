"""Health scan, tested against a model with KNOWN planted pathologies.

Two real bugs are pinned here:
  * duplicates were first computed from each channel's mean MAGNITUDE, which
    reported 54 duplicate pairs in a layer that had exactly 1 — channels with
    similar average energy look identical by that measure. Only the activation
    PATTERN distinguishes them.
  * Flatten/Reshape were reported as layers, inventing ~33 phantom "dead units"
    in a perfectly healthy model. They're views of the previous layer, not
    units of their own.
"""

import pytest

from conftest import build_sabotaged

DEAD = (0, 1, 2)
DUP = (5, 6)


@pytest.fixture(scope="module")
def sick(client):
    return build_sabotaged(client, dead=DEAD, dup=DUP)


@pytest.fixture(scope="module")
def sick_scan(client, sick):
    r = client.get(f"/api/models/{sick}/health", params={"n": 20})
    assert r.status_code == 200
    return r.json()


@pytest.fixture(scope="module")
def healthy_scan(client, cnn):
    r = client.get(f"/api/models/{cnn}/health", params={"n": 20})
    assert r.status_code == 200
    return r.json()


def _layer(scan, node):
    return next(r for r in scan["profile"] if r["node"] == node)


def test_finds_exactly_the_planted_dead_channels(sick_scan):
    assert _layer(sick_scan, "relu1")["dead"] == len(DEAD)


def test_dead_channels_propagate_through_pooling(sick_scan):
    # pooling a channel that is always zero keeps it always zero
    assert _layer(sick_scan, "pool1")["dead"] == len(DEAD)


def test_finds_exactly_one_duplicate_pair_and_names_it(sick_scan):
    """Regression: magnitude-based correlation reported 54 pairs here."""
    relu1 = _layer(sick_scan, "relu1")
    assert relu1["dup_pairs"] == 1
    a, b, corr = relu1["dup_examples"][0]
    assert {a, b} == set(DUP)
    assert corr > 0.999


def test_healthy_model_reports_no_dead_units(healthy_scan):
    """Regression: Flatten used to invent ~33 phantom dead units here."""
    assert healthy_scan["totals"]["dead"] == 0


def test_healthy_model_reports_no_duplicates(healthy_scan):
    assert healthy_scan["totals"]["dup"] == 0


def test_structural_ops_are_not_reported_as_layers(healthy_scan):
    ops = {r["op"] for r in healthy_scan["profile"]}
    assert not ops & {"Flatten", "Reshape", "Transpose", "Squeeze"}


def test_sick_model_is_separable_from_healthy(sick_scan, healthy_scan):
    assert sick_scan["totals"]["dead"] > healthy_scan["totals"]["dead"]


def test_scan_covers_the_computing_layers(healthy_scan):
    assert healthy_scan["totals"]["layers"] >= 6
    assert healthy_scan["totals"]["runs"] >= 3


def test_refuses_too_few_probes_gracefully(client, cnn):
    r = client.get(f"/api/models/{cnn}/health", params={"n": 4})
    assert r.status_code == 200


def test_health_scan_works_on_text_models(client, transformer):
    r = client.get(f"/api/models/{transformer}/health", params={"n": 8})
    assert r.status_code == 200
    assert r.json()["totals"]["layers"] > 0
