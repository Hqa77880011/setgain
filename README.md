# SetGain

Implementation of **SetGain: Learning the Marginal Value of Additional Annotations for Ambiguous Medical Segmentation**.

## Installation

Use Python `>=3.10`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

Obtain [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) DICOM/XML annotations. Install the LIDC extra and configure its DICOM path using the [pylidc guide](https://pylidc.github.io/install.html). Preparation keeps matched nodules with at least four individual contours and splits by patient (seed 2027).

```bash
python -m pip install -e ".[lidc]"
setgain prepare --source lidc --output data/lidc
```

For [QUBIQ](https://qubiq.grand-challenge.org/participation/), use `setgain prepare --source qubiq --input data/qubiq/source_mapping.json --output data/qubiq/prepared`. Mapping records contain `id,patient_id,task,image,labels` and an image-selected `slice` for 3D cases; individual readers remain separate. NPZ manifests are also supported with `--source manifest`, arrays `image,masks,spacing`, and stable patient IDs.

## Training

Fit the conditional predictor before constructing disjoint visible/candidate/hidden-reader gain targets. Training and target settings are in `configs/default.json`; keep `--seed 17` paired across these stages.

```bash
setgain train-predictor --manifest data/lidc/manifest.json --output checkpoints/predictor --seed 17
setgain make-targets --manifest data/lidc/manifest.json --checkpoint checkpoints/predictor/predictor.pt --seed 17 --output data/gains.npz
setgain train-gain --targets data/gains.npz --seed 17 --output checkpoints/gain.pt
```

## Inference and evaluation

The explicit acquisition seeds pair the reveal orders across policies.

```bash
setgain run --manifest data/lidc/manifest.json --checkpoint checkpoints/predictor/predictor.pt --head checkpoints/gain.pt --policy setgain --seed 17 --seeds 100 101 102 103 104 105 106 107 108 109 --output runs/setgain
setgain evaluate --inputs runs/setgain/results.json --output runs/comparison.json
setgain plot --input runs/comparison.json --output runs/figures
```

## Ablations

`no_count`, `no_disagreement`, `dice_target` and `shuffled` need matching target construction, fitted head and run policy. `no_exploration` reuses the full head. `no_set` needs a separately trained `train-predictor --no-set` predictor and its own targets/head.

```bash
setgain make-targets --manifest data/lidc/manifest.json --checkpoint checkpoints/predictor/predictor.pt --variant no_count --seed 17 --output data/no_count.npz
setgain train-gain --targets data/no_count.npz --variant no_count --seed 17 --output checkpoints/no_count.pt
setgain run --manifest data/lidc/manifest.json --checkpoint checkpoints/predictor/predictor.pt --head checkpoints/no_count.pt --policy no_count --seed 17 --output runs/no_count
setgain evaluate --inputs runs/setgain/results.json runs/no_count/results.json --output runs/comparison.json
```

## Outputs

Predictor fitting saves `predictor.pt` and `history.json`; target construction writes the NPZ and its JSON reader-partition companion. Gain fitting saves its checkpoint and JSON history. Runs save `results.json` and `calibration.json` with query, budget and timing records. Evaluation writes summary JSON; figures are PNG/SVG.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
