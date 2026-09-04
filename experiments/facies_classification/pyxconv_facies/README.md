# pyxconv_facies

Local facies fork of `pyxconv`. **Do not** edit the conda `site-packages/pyxconv` install.

## Source

Copied from [`xconv_pv`](../../xconv_pv) on branch **`ali`** (commit `70825e2` at time of copy).

Upstream changes on `ali` vs older copies include padding-aware probe shifts (`_shift2d_A`), `padding_mode` support on `Xconv2D`/`Xconv3D`, and relative imports in `modules.py` / `funcs.py`.

## Facies additions

| File | Purpose |
|------|---------|
| `facies_convert.py` | `apply_xconv_to_facies_model`, layer counts, plot title helper |
| `funcs.py` | `XconvTranspose2D` probing backward (uses `pyxconv.probe`) |
| `modules.py` | `XconvTranspose2D` module |
| `utils.py` | `adaptive_convert_facies`, `align_grad_output_to_input_spatial` |
| `probe.py` | Re-exports `pyxconv.probe` (do not duplicate probe kernels locally) |

`apply_xconv_to_facies_model` converts **Conv2d** and **ConvTranspose2d** when **H×W > ps** and **channels < 512** (skips fc6/fc7 on `patch_deconvnet`).

## Usage

```python
from pyxconv_facies.facies_convert import apply_xconv_to_facies_model

apply_xconv_to_facies_model(model, sample_images, probing_vector=16, xmode='independent')
```

Run from repo root with `PYTHONPATH=.` or install as editable if preferred.

## Tests

```bash
conda activate facies_classn_env
cd /path/to/facies_classification_benchmark_changes
PYTHONPATH=. python pyxconv_facies/test_pyxconv_facies.py
```
