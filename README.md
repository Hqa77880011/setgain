# SetGain

Implementation of **SetGain: Learning the Marginal Value of Additional Annotations for Ambiguous Medical Segmentation**. The model predicts a distribution of masks conditioned on an anonymous visible reader set. A gain head learns the decrease in hidden-reader energy score after acquiring another annotation and allocates an extra-mask budget across cases.

## Installation and data

Use Python 3.11 with PyTorch appropriate for your hardware:

```bash
python -m venv .venv
python -m pip install -e .
```

Activate the environment before installation. Commands run from the repository root and choose CUDA when available; use `--device cpu` otherwise.

Obtain [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) DICOM and XML files. Install `pylidc`, configure its DICOM path following the [installation guide](https://pylidc.github.io/install.html), and inspect ambiguous nodule associations. The converter retains matched nodules with at least four individually delineated contours, stores full contour volumes, and splits by patient before training.

```bash
python -m pip install -e ".[lidc]"
setgain prepare --source lidc --output data/lidc
```

One visible initial contour determines the largest-area slice and the 128 mm square crop. This field is held fixed throughout its acquisition episode; later and hidden contours do not set its geometry. The resulting 128 × 128 field has 1 mm in-plane spacing. Local reader indices denote anonymous annotations.

For [QUBIQ](https://qubiq.grand-challenge.org/participation/), use individually annotated brain-growth, prostate and kidney cases. Export each task as a NPZ with `image` (H × W), `masks` (readers × H × W) and `spacing` (two millimetre values). A JSON manifest contains `id`, `patient_id`, `dataset: "QUBIQ"`, `task`, `path`, and `modality`. Preserve the patient ID across prostate structures. Select slices from public image geometry or a fixed curator protocol; do not use hidden masks to choose fields. Exclude adult-glioma fused labels from the human-reader cohort.

```bash
setgain prepare --source manifest --input data/qubiq/cases.json --output data/qubiq/manifest.json
```

For native NIfTI packs, `setgain prepare --source qubiq --input data/qubiq/source_mapping.json --output data/qubiq/prepared` converts aligned reader files directly. Mapping entries contain `id`, `patient_id`, `task` (`brain-growth`, `prostate`, `kidney`), `image`, `labels` (individual reader paths), optional `structure_label`, and an image-selected `slice` for 3D data. It checks affine alignment and preserves both prostate structures under the same patient ID.

The import checks binary masks and image alignment, assigns patient-level 60/20/20 partitions with seed 2027, and resolves paths relative to the source manifest. QUBIQ uses 256 × 256 full fields. Combine manifests before preparation to evaluate a multi-task cohort. Training patients are further separated into two thirds for predictor fitting and one third for gain-target construction; saved predictor checkpoints retain the fitted patient IDs.

## Train the conditional predictor

```bash
setgain train-predictor --manifest data/lidc/manifest.json --output checkpoints/predictor --seed 17
```

The four-level U-Net uses widths 32/64/128/256, an eight-dimensional Gaussian latent and 128-dimensional contour embeddings. Symmetric set conditioning concatenates the first and second embedding moments and log(1 + count). The variational objective combines normalized Bernoulli likelihood and a KL term of weight 0.001. A reader outside the visible set supplies each training target. The validation hidden-reader energy score selects `predictor.pt`.

## Construct gains and fit the acquisition head

```bash
setgain make-targets --manifest data/lidc/manifest.json --checkpoint checkpoints/predictor/predictor.pt --output data/gains.npz
setgain train-gain --targets data/gains.npz --output checkpoints/gain.pt
```

Each gain example has disjoint visible, candidate and hidden readers. Before/after distributions reuse independent base Gaussian and Bernoulli draws across states, preserve negative gains, and use 16 samples. The head sees image/set features, remaining count, visible disagreement statistics and prior log-variance. Candidate and hidden masks enter target construction only. Gain examples and reader partitions are stored in `gains.npz` and its JSON companion. The squared-error head is trained for 100 epochs; its checkpoint is `gain.pt`.

`configs/default.json` controls epoch counts, learning rates, batch sizes, latent-sample count, partitions per gain-fit case, exploration and the budget grid. Predictor defaults use 200 epochs and Adam at 0.0001. Gain fitting uses Adam at 0.001. `--config` selects an alternate file. Use seeds 17, 29 and 43 for independently fitted predictors and heads.

## Budgeted acquisition and metrics

```bash
setgain run --manifest data/lidc/manifest.json --checkpoint checkpoints/predictor/predictor.pt --head checkpoints/gain.pt --policy setgain --output runs/setgain
setgain evaluate --inputs runs/setgain/results.json --output runs/comparison.json
setgain plot --input runs/comparison.json --output runs/figures
setgain plot --input runs/setgain/calibration.json --output runs/gain_figures
```

Every episode holds one reader out permanently and starts with one other reader. The policy chooses a case and receives the next reader in a precommitted random reveal order. It uses the entire fixed budget even if all predicted gains are negative. Evaluation rotates every possible held-out reader and pairs seeds 100–109 across policies. The extra-mask pool excludes initial and permanently hidden masks.

`results.json` contains per-budget summaries, patient/task aggregation, query records, gain calibration bins and measured scoring/update times. Its metrics cover energy score, normalized trapezoidal AURC, Brier score, ten-bin ECE, minority-mask coverage, disagreement gap, gain MAE, Spearman correlation and the negative-gain query fraction. Energy score uses unsquared Euclidean mask distance divided by √pixel-count and distinct sample pairs with denominator M(M−1). Minority groups use complete-linkage Dice-distance clustering at 0.20, exclusively in evaluation. Coverage is best-of-16 Dice against minority hidden masks. Nodules are averaged within patients before equal task aggregation.

## Baselines and component removals

Run `random`, `uniform`, `fewest`, `disagreement`, `entropy`, `diversity_entropy`, `current_loss`, `feature_lal`, `bald`, `epig`, `empirical` or `oracle` through the same `run` command. `uniform` and `fewest` share tie order. BALD and EPIG need three independently trained auxiliary predictors via `--ensemble checkpoint1 checkpoint2 checkpoint3`; emitted masks always come from the primary predictor. Feature LAL needs `--targets data/gains.npz` and fits a 300-tree depth-eight regressor. `current_loss` requires a separate head trained with `train-gain --variant current_loss`. The empirical control mixes visible masks with one learned image/set prior component. `oracle` reads sealed evaluator masks for a privileged diagnostic.

```bash
setgain run --manifest data/lidc/manifest.json --checkpoint checkpoints/predictor/predictor.pt --policy uniform --output runs/uniform
setgain evaluate --inputs runs/setgain/results.json runs/uniform/results.json --output runs/comparison.json
```

For `no_count`, `no_disagreement`, `dice_target` and `shuffled`, generate targets with the matching `make-targets --variant`, fit a matching gain head, then use that policy. `no_exploration` reuses the full gain head. `no_set` needs a predictor fitted with `train-predictor --no-set`, followed by gain construction and fitting on that predictor. The consensus target uses the disjoint target-reader set rather than labels visible to the head.

The implementation uses finite recorded-reader pools and fixed two-dimensional fields. The empirical mixture includes a learned prior rather than an externally pretrained model. Timing records the candidate pool size and measured scoring, update and emission costs. These conventions determine how the resulting tables should be read.

## Complete analysis grid

Use one prepared manifest for each domain and repeat fitting with seeds 17, 29 and 43. `gain_passes=25` and `partitions_per_case=4` produce 100 disjoint reader partitions per eligible gain-fit case. Shuffled targets are permuted within visible-count strata. `no_count` removes both visible and remaining counts from the gain input; the shared predictor is retained.

| Analysis | Inputs or policy | Output |
| --- | --- | --- |
| Main budget curve | Every listed policy, budgets 0/0.25/0.5/0.75/1 | ES, normalized AURC, Brier, ECE, MMC and disagreement gap |
| Components | Matching predictor/target/head for each removal | The same budget metrics |
| Gain prediction | `setgain`, `feature_lal`, `shuffled` | MAE, Spearman, negative-gain fraction and calibration bins |
| Boundary reliability | Included in `run` | Brier and ECE in a ten-pixel band around the initial visible contour |
| Cached efficiency | Add `--score-states 1000` to `run` | Scoring, selected-case update, 16-mask emission and their summed time |

The scoring benchmark cycles through the available candidate states to reach the requested count and records the unique case count. CUDA timers synchronize before recording elapsed times. Reader rotation uses the least common multiple of the cohort's reader counts; every case remains in every rotation and every local reader is held out equally often. Calibration plots use predicted gain on the horizontal axis. Figures are saved as PNG and SVG.

For a component removal, replace `no_count` below with `no_disagreement`, `dice_target` or `shuffled`:

```bash
setgain make-targets --manifest data/lidc/manifest.json --checkpoint checkpoints/predictor/predictor.pt --variant no_count --output data/no_count.npz
setgain train-gain --targets data/no_count.npz --variant no_count --output checkpoints/no_count.pt
setgain run --manifest data/lidc/manifest.json --checkpoint checkpoints/predictor/predictor.pt --head checkpoints/no_count.pt --policy no_count --output runs/no_count
```

For `no_set`, train with `--no-set`, then construct targets and fit the gain head using that predictor before running `--policy no_set`. For `current_loss`, use `train-gain --variant current_loss` with the full target archive.
