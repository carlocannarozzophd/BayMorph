"""Fixtures shared by several test modules."""

from __future__ import annotations

import os
import urllib.request
from copy import deepcopy

import numpy as np
import pytest

from baymorph.pipeline import calibration_bridges
from baymorph.rendering.effective_psf import SUBPIXEL_FACTOR
from baymorph.simulation.psf import EmpiricalPSF

# The continuous-integration machines have no LaTeX, so the tests draw figures
# with Matplotlib's Computer Modern mathtext unless BAYMORPH_USETEX is already
# set: BAYMORPH_USETEX=1 runs them with LaTeX.
os.environ.setdefault("BAYMORPH_USETEX", "0")

ACS_ANSWER_HEADERS = [
    "Detector",
    "Filter",
    "PHOTLAM [Å]",
    "PHOTFLAM [erg cm⁻² Å⁻¹ e⁻¹]",
    "STmag",
    "VEGAmag",
    "ABmag",
]

ACS_ANSWERS = {
    "2007-01-22": {
        "headers": ACS_ANSWER_HEADERS,
        "rows": [["WFC", "F814W", "8045.2", "7.1337e-20", "26.767", "25.500", "25.931"]],
    },
    "2004-04-15": {
        "headers": ACS_ANSWER_HEADERS,
        "rows": [["WFC", "F814W", "8047.4", "6.9951e-20", "26.788", "25.520", "25.952"]],
    },
    "2006-06-20": {
        "headers": ACS_ANSWER_HEADERS,
        "rows": [["WFC", "F814W", "8048.0", "7.0186e-20", "26.784", "25.517", "25.948"]],
    },
    "2006-07-20": {
        "headers": ACS_ANSWER_HEADERS,
        "rows": [["WFC", "F814W", "8045.1", "7.1281e-20", "26.768", "25.501", "25.932"]],
    },
}
"""Answers of the ACS Zeropoint Calculator of STScI to WFC F814W, received on
2026-10-20."""

ANSWER_TIME = "2026-10-20T08:15:02Z"


@pytest.fixture(autouse=True)
def _no_calibration_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that would reach a calibration service on the network.

    The requests of the bridges are refused, and so is any request that
    urllib would send over HTTP or HTTPS; a test of the transport replaces
    the second guard with its own answers.
    """

    def refuse(url, body, *, timeout):
        pytest.fail(f"A test tried to reach a calibration service: {url}")

    def refuse_transport(handler, request):
        pytest.fail(f"A test tried to send a request: {request.full_url}")

    monkeypatch.setattr(calibration_bridges, "post_json", refuse)
    monkeypatch.setattr(calibration_bridges, "_utc_now", lambda: ANSWER_TIME)
    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", refuse_transport)
    monkeypatch.setattr(urllib.request.HTTPHandler, "http_open", refuse_transport)


class FakeCalculator:
    """Answer the requests of the calibration bridges as the calculator did.

    ``calls`` records each request; ``answer`` replaces the answer of every
    date, ``by_date`` adds answers for other dates, and ``error`` is raised
    instead of answering. ``answers`` holds the answers received from the
    calculator by date, and ``asked_at`` the time every test reads.
    """

    answers = ACS_ANSWERS
    asked_at = ANSWER_TIME

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, float]] = []
        self.answer: object | None = None
        self.by_date: dict[str, object] = {}
        self.error: Exception | None = None

    def __call__(self, url, body, *, timeout):
        self.calls.append((url, dict(body), timeout))

        if self.error is not None:
            raise self.error

        if self.answer is not None:
            return deepcopy(self.answer)

        if body["date"] in self.by_date:
            return deepcopy(self.by_date[body["date"]])

        return deepcopy(ACS_ANSWERS[body["date"]])


@pytest.fixture
def acs_calculator(monkeypatch: pytest.MonkeyPatch) -> FakeCalculator:
    """Answer as the ACS Zeropoint Calculator, without the network."""
    fake = FakeCalculator()
    monkeypatch.setattr(calibration_bridges, "post_json", fake)

    return fake


def gaussian_subpixel_psf(sigma: float = 1.1, half: int = 7) -> EmpiricalPSF:
    """Return a smooth empirical PSF with kernels at every sub-pixel phase.

    The kernels sample a circular Gaussian at the pixel centres displaced by
    each phase, normalized to unit sum: a valid sub-pixel product for tests
    of placement and rendering, not a pixel-integrated PSF.
    """
    factor = SUBPIXEL_FACTOR
    centre = (factor - 1) // 2
    offsets = np.arange(2 * half + 1, dtype=float) - half
    kernels = np.empty((factor, factor, 2 * half + 1, 2 * half + 1))

    for j in range(factor):
        for i in range(factor):
            x = offsets + (i - centre) / factor
            y = offsets + (j - centre) / factor
            kernel = np.exp(
                -(x[None, :] ** 2 + y[:, None] ** 2) / (2.0 * sigma**2)
            )
            kernels[j, i] = kernel / kernel.sum()

    return EmpiricalPSF(kernels[centre, centre], subpixel_kernels=kernels)


@pytest.fixture
def subpixel_gaussian_psf() -> EmpiricalPSF:
    """Return :func:`gaussian_subpixel_psf` with its default width."""
    return gaussian_subpixel_psf()
