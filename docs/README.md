# BayMorph Documentation

BayMorph is a configuration-driven Python package for preparing astronomical images and fitting galaxy morphology with Bayesian models.

## Core design

BayMorph follows a strict separation of responsibilities:

```text
notebooks
    ↓
BayMorphRunner and public workflows
    ↓
pipeline orchestration
    ↓
core scientific modules
```

Notebooks should contain only configuration loading, workflow calls, inspection, and interpretation. Detection, masking, cutout resolution, background estimation, fitting, and diagnostics belong in the library.

## Installation and updates

### User installation

```bash
python -m pip install baymorph
```

Update BayMorph with:

```bash
python -m pip install --upgrade baymorph
```

### Repository installation

```bash
git clone https://github.com/carlocannarozzophd/BayMorph.git
cd BayMorph
python -m pip install .
```

### Development installation

```bash
python -m pip install -e ".[dev]"
```

BayMorph requires Python 3.10 or newer.

## Creating an external project

After installation, create a working directory outside `site-packages`:

```bash
baymorph init MyProject
```

The command creates:

```text
MyProject/
├── configs/
├── notebooks/
├── data/
└── runs/
```

Alternatively, copy only the packaged examples:

```bash
baymorph examples BayMorphExamples
```

The copied notebooks and YAML files are normal user files. Updating the installed package does not overwrite them.

## Standard workflow

```python
from baymorph import BayMorphRunner, load_resolved_config

config = load_resolved_config("configs/real_galaxy.yaml")
runner = BayMorphRunner.from_config(config)

runner.prepare_image()
runner.select_psf_candidates()
runner.build_psf()
runner.fit()
runner.diagnostics()
```

For a simulated workflow with a configured analytic PSF, candidate selection and empirical PSF construction may be unnecessary:

```python
runner.prepare_image()
runner.fit()
runner.diagnostics()
```

Products are exposed as:

```python
runner.image
runner.psf_candidates
runner.psf
runner.results
```

## Image preparation

The main target-specific workflow is implemented by `prepare_real_image_from_config()` and exposed through `runner.prepare_image()`.

### Fixed cutout mode

```yaml
data:
  cutout:
    mode: fixed
    size: 200
```

### Automatic cutout mode

```yaml
data:
  cutout:
    mode: auto
    initial_factor: 4.0
    sky_margin: 20.0
    min_size: 100
    max_size: null
```

The automatic mode performs:

1. detection on the full image;
2. target selection near the configured seed position;
3. a preliminary adaptive cutout from the target-segment bounding box;
4. a second detection in the preliminary cutout;
5. target-geometry estimation from second moments;
6. elliptical radial-profile measurement;
7. plateau detection;
8. final cutout resolution from the plateau radius and sky margin;
9. final detection and contaminant masking;
10. final geometry, profile, plateau, and sky-region construction;
11. constant-background subtraction and robust noise-map construction.

### Position semantics

`data.target_position` is an initial target-identification seed in full-image coordinates. The final target centre stored in `ImagePreparationProducts.target_x` and `.target_y` is the measured geometric centre in final-cutout coordinates.

### Background and noise

The final constant background and robust noise estimate are measured from the automatically constructed sky region outside the detected target plateau, excluding contaminant pixels. This avoids inflating the noise estimate with target-galaxy light.

## Main products

`ImagePreparationProducts` includes:

- original image;
- final cutout and bounds;
- prepared background-subtracted data;
- noise map;
- final segmentation;
- target segment;
- contaminant mask;
- measured target centre;
- final geometry;
- final radial profile;
- final plateau result;
- final sky region;
- preliminary cutout and preliminary segmentation for automatic mode.

`RealFitProducts` includes:

- model name;
- best parameters;
- posterior result;
- best-model image;
- residual and normalized residual;
- chi-square statistics;
- radial-profile products;
- generated diagnostic handles or paths where applicable.

## Configuration

Use the packaged reference configuration as the complete option catalogue:

```text
configs/config_reference.yaml
```

Minimal user configurations may omit fields filled by BayMorph defaults.

## CLI

```bash
baymorph --help
baymorph --version
baymorph examples DESTINATION
baymorph init PROJECT_DIRECTORY
```

Both `examples` and `init` support `--force` for explicit overwriting.

## Testing

Run:

```bash
ruff check .
pytest
```

The test suite covers core numerics, source detection, sky geometry, automatic cutouts, pipeline orchestration, fitting, plotting, CLI behaviour, and serialization.

## Further documentation

- [Architecture](ARCHITECTURE.md)
- [Validation](VALIDATION.md)
- [Architectural decisions](../DECISIONS.md)
- [Roadmap](../ROADMAP.md)
