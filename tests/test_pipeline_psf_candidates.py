from baymorph import (
    PSFCandidate,
    load_psf_candidates,
    pipeline_paths,
    resolve_config,
    save_psf_candidates,
)


def test_save_and_load_psf_candidates_roundtrip(tmp_path):
    config = {
        "object": {"name": "NGC1234"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": str(tmp_path / "outputs")},
    }

    resolved = resolve_config(config)

    candidates = [
        PSFCandidate(x=10.0, y=20.0, flux=1000.0, fwhm=2.5),
        PSFCandidate(x=30.0, y=40.0, flux=900.0, fwhm=2.7),
    ]

    path = save_psf_candidates(resolved, candidates)

    assert path == pipeline_paths(resolved)["psf"]["candidates"]
    assert path.exists()

    loaded = load_psf_candidates(resolved)

    assert loaded == candidates