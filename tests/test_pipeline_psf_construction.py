import numpy as np

from baymorph import (
    load_empirical_psf,
    pipeline_paths,
    resolve_config,
    save_empirical_psf,
)


def test_save_and_load_empirical_psf_roundtrip(tmp_path):
    config = {
        "object": {"name": "NGC1234"},
        "data": {"image_path": "data/image.fits"},
        "outputs": {"root": str(tmp_path / "outputs"), "overwrite": True},
    }

    resolved = resolve_config(config)

    psf = np.ones((9, 9))
    psf /= psf.sum()

    path = save_empirical_psf(resolved, psf)

    assert path == pipeline_paths(resolved)["psf"]["psf_fits"]
    assert path.exists()

    loaded = load_empirical_psf(resolved)

    assert np.allclose(loaded, psf)