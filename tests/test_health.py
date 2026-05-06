from __future__ import annotations

import pytest

from rtpfb.errors import HealthCheckFailed
from rtpfb.health import (
    CheckResult,
    HealthReport,
    check_face_image,
    check_swap_model,
    check_third_party,
)


def test_report_ok_when_all_pass():
    r = HealthReport()
    r.add(CheckResult("a", True))
    r.add(CheckResult("b", True))
    assert r.ok() is True
    assert r.failures() == []


def test_report_fails_with_details():
    r = HealthReport()
    r.add(CheckResult("a", True))
    r.add(CheckResult("b", False, "missing X"))
    assert r.ok() is False
    assert r.failures() == ["b: missing X"]


def test_raise_if_failed_aggregates_message():
    r = HealthReport()
    r.add(CheckResult("a", False, "x"))
    r.add(CheckResult("b", False, "y"))
    with pytest.raises(HealthCheckFailed) as ei:
        r.raise_if_failed()
    msg = str(ei.value)
    assert "a: x" in msg
    assert "b: y" in msg


def test_check_face_image_missing(tmp_path):
    res = check_face_image(str(tmp_path / "nope.png"))
    assert not res.ok


def test_check_face_image_present(tmp_path):
    p = tmp_path / "face.png"
    p.write_bytes(b"x")
    res = check_face_image(str(p))
    assert res.ok


def test_check_face_image_empty_path():
    res = check_face_image("")
    assert not res.ok


def test_check_swap_model_missing():
    res = check_swap_model("definitely-not-a-real-model.onnx")
    assert not res.ok


def test_check_third_party_missing():
    res = check_third_party("not-a-real-repo")
    assert not res.ok
