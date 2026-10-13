import argparse
import heapq
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from scipy.integrate import trapezoid
from sklearn.ensemble import RandomForestRegressor

from .data import fixed_field, prepare_lidc, prepare_manifest, records
from .method import (
    ConditionalModel,
    GainHead,
    POLICIES,
    dice,
    energy_score,
    pair_distance,
    predictive_metrics,
)
from .utils import load, peak_memory_mib, plot_results, save, seed_all


def field(row, initial=0):
    return fixed_field(
        row, initial=initial, size=128 if row["dataset"] == "LIDC" else 256, side_mm=128
    )[:2]


def image_tensor(image, device):
    return torch.as_tensor(image, dtype=torch.float32, device=device)[None, None]


def checkpoint_model(path, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    model = ConditionalModel(no_set=saved.get("no_set", False)).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model


def train_predictor(args):
    cfg = load(args.config)
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    rows = records(args.manifest, "train")
    patients = sorted({r["patient_id"] for r in rows})
    patients = np.random.default_rng(2027).permutation(patients).tolist()
    selected = set(patients[: int(2 * len(patients) / 3)])
    rows = [r for r in rows if r["patient_id"] in selected]
    if not rows or len(selected) == len(patients):
        raise ValueError("Need separate backbone-fit and gain-fit training patients")
    model = ConditionalModel(no_set=args.no_set).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["predictor_lr"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    for epoch in range(cfg["predictor_epochs"]):
        model.train()
        total, count = 0.0, 0
        optimizer.zero_grad()
        for index in rng.permutation(len(rows)):
            row = rows[index]
            with np.load(row["path"], allow_pickle=False) as obj:
                readers = len(obj["masks"])
            order = rng.permutation(readers)
            target_index, first = int(order[0]), int(order[1])
            image, masks = field(row, first)
            if rng.random() < 0.5:
                image, masks = image[:, ::-1].copy(), masks[:, :, ::-1].copy()
            rotation = int(rng.integers(4))
            image, masks = (
                np.rot90(image, rotation).copy(),
                np.rot90(masks, rotation, axes=(1, 2)).copy(),
            )
            visible = order[1 : 1 + int(rng.integers(1, readers))]
            loss = model.loss(
                image_tensor(image, args.device),
                torch.as_tensor(masks[visible], device=args.device),
                torch.as_tensor(masks[target_index], device=args.device),
                cfg["beta"],
            )
            (loss / cfg["batch_size"]).backward()
            total += float(loss.detach())
            count += 1
            if count % cfg["batch_size"] == 0:
                optimizer.step()
                optimizer.zero_grad()
        if count % cfg["batch_size"]:
            scale = cfg["batch_size"] / (count % cfg["batch_size"])
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            optimizer.step()
        model.eval()
        risk = []
        with torch.inference_mode():
            torch.manual_seed(2027)
            for row in records(args.manifest, "validation"):
                image, masks = field(row, 0)
                x = image_tensor(image, args.device)
                features = model.encode_image(x)
                state = model.statistics(
                    model.embed_masks(x, torch.as_tensor(masks[:1], device=args.device))
                )
                normal = torch.randn(cfg["samples"], 8, device=args.device)
                uniforms = torch.rand(cfg["samples"], *image.shape, device=args.device)
                samples, _ = model.sample(features, state, normal, uniforms)
                risk.append(energy_score(samples, masks[1:]))
        score = float(np.mean(risk))
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total / max(count, 1),
                "validation_energy_score": score,
            }
        )
        if score < best:
            best = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "no_set": args.no_set,
                    "seed": args.seed,
                    "config": cfg,
                    "fit_patients": sorted(selected),
                },
                output / "predictor.pt",
            )
        save(output / "history.json", history)
        print(f"Epoch {epoch + 1}: validation energy score={score:.6f}", flush=True)


def make_targets(args):
    seed_all(args.seed)
    cfg, model = load(args.config), checkpoint_model(args.checkpoint, args.device)
    fit_patients = set(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)[
            "fit_patients"
        ]
    )
    rows = [
        r
        for r in records(args.manifest, "train")
        if r["patient_id"] not in fit_patients
    ]
    rng = np.random.default_rng(args.seed)
    inputs, targets, losses, metadata = [], [], [], []
    with torch.inference_mode():
        for row in rows:
            with np.load(row["path"], allow_pickle=False) as obj:
                number = len(obj["masks"])
            if number < 3:
                continue
            for _ in range(cfg["partitions_per_case"] * cfg.get("gain_passes", 25)):
                order = rng.permutation(number)
                hidden, candidate, initial = map(int, order[:3])
                image, masks = field(row, initial)
                extra = int(rng.integers(number - 2))
                visible = [initial] + list(map(int, order[3 : 3 + extra]))
                hidden_indices = [hidden]
                x = image_tensor(image, args.device)
                features = model.encode_image(x)
                embeddings = model.embed_masks(
                    x, torch.as_tensor(masks[visible], device=args.device)
                )
                before = model.statistics(embeddings)
                after = model.statistics(
                    torch.cat(
                        [
                            embeddings,
                            model.embed_masks(
                                x,
                                torch.as_tensor(
                                    masks[candidate : candidate + 1], device=args.device
                                ),
                            ),
                        ]
                    )
                )
                normal = torch.randn(cfg["samples"], 8, device=args.device)
                uniforms = torch.rand(cfg["samples"], *image.shape, device=args.device)
                old, _ = model.sample(features, before, normal, uniforms)
                new, _ = model.sample(features, after, normal, uniforms)
                risk_before, risk_after = (
                    energy_score(old, masks[hidden_indices]),
                    energy_score(new, masks[hidden_indices]),
                )
                value = risk_before - risk_after
                if args.variant == "dice_target":
                    consensus = masks[hidden_indices].mean(0) >= 0.5
                    value = np.mean([dice(s, consensus) for s in new]) - np.mean(
                        [dice(s, consensus) for s in old]
                    )
                vector = (
                    model.gain_features(
                        features,
                        before,
                        torch.as_tensor(masks[visible], device=args.device),
                        number - len(visible) - len(hidden_indices),
                        args.variant,
                    )
                    .cpu()
                    .numpy()
                )
                inputs.append(vector)
                targets.append(value)
                losses.append(risk_before)
                metadata.append(
                    {
                        "id": row["id"],
                        "patient_id": row["patient_id"],
                        "visible": visible,
                        "candidate": candidate,
                        "hidden": hidden_indices,
                    }
                )
    if not inputs:
        raise ValueError("No gain examples with three disjoint reader groups")
    targets = np.asarray(targets, dtype=np.float32)
    if args.variant == "shuffled":
        counts = np.asarray([len(r["visible"]) for r in metadata])
        for count in np.unique(counts):
            indices = np.flatnonzero(counts == count)
            targets[indices] = rng.permutation(targets[indices])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        inputs=np.asarray(inputs, dtype=np.float32),
        targets=targets,
        current_loss=np.asarray(losses, dtype=np.float32),
    )
    save(output.with_suffix(".json"), {"variant": args.variant, "examples": metadata})


def train_gain(args):
    cfg = load(args.config)
    seed_all(args.seed)
    with np.load(args.targets, allow_pickle=False) as obj:
        x = torch.as_tensor(obj["inputs"], device=args.device)
        y = torch.as_tensor(
            obj["current_loss" if args.variant == "current_loss" else "targets"],
            device=args.device,
        )[:, None]
    head = GainHead().to(args.device)
    optimizer = torch.optim.Adam(head.parameters(), lr=cfg["gain_lr"])
    history = []
    for epoch in range(cfg["gain_epochs"]):
        values = []
        for indices in torch.randperm(len(x), device=args.device).split(
            cfg["gain_batch_size"]
        ):
            optimizer.zero_grad()
            loss = (head(x[indices]) - y[indices]).square().mean()
            loss.backward()
            optimizer.step()
            values.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "mse": float(np.mean(values))})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": head.state_dict(), "variant": args.variant, "config": cfg}, path
    )
    save(path.with_suffix(".json"), history)


def weighted_mean(items):
    patients = defaultdict(list)
    for item in items:
        patients[
            (item["dataset"], item.get("task", item["dataset"]), item["patient_id"])
        ].append(item["metrics"])
    tasks = defaultdict(list)
    keys = (
        "energy_score",
        "brier",
        "ece",
        "minority_mask_coverage",
        "disagreement_gap",
        "band_brier",
        "band_ece",
    )
    for (dataset, task, _), metrics in patients.items():
        row = {
            key: float(np.mean([r[key] for r in metrics if r[key] is not None]))
            if any(r[key] is not None for r in metrics)
            else None
            for key in keys
        }
        tasks[(dataset, task)].append(row)
    task_means = [
        {
            key: float(np.mean([r[key] for r in values if r[key] is not None]))
            if any(r[key] is not None for r in values)
            else None
            for key in keys
        }
        for values in tasks.values()
    ]
    return {
        key: float(np.mean([r[key] for r in task_means if r[key] is not None]))
        if any(r[key] is not None for r in task_means)
        else None
        for key in keys
    }


def run(args):
    cfg = load(args.config)

    def sync():
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()

    model = checkpoint_model(args.checkpoint, args.device)
    head = GainHead().to(args.device)
    if args.head:
        head.load_state_dict(
            torch.load(args.head, map_location=args.device, weights_only=True)["model"]
        )
    elif args.policy in (
        "setgain",
        "no_count",
        "no_disagreement",
        "dice_target",
        "shuffled",
        "no_exploration",
        "no_set",
        "current_loss",
    ):
        raise ValueError("This policy requires --head")
    head.eval()
    lal = None
    if args.policy == "feature_lal":
        if not args.targets:
            raise ValueError("feature_lal requires --targets")
        with np.load(args.targets, allow_pickle=False) as obj:
            lal = RandomForestRegressor(
                n_estimators=300, max_depth=8, random_state=17
            ).fit(obj["inputs"][:, -13:], obj["targets"])
    ensemble = [checkpoint_model(p, args.device) for p in args.ensemble]
    if args.policy in ("bald", "epig") and len(ensemble) != 3:
        raise ValueError(
            "BALD and EPIG require exactly three auxiliary --ensemble checkpoints"
        )
    rows = records(args.manifest, args.split)
    repeats, query_rows, timing = [], [], []
    reader_counts = []
    for row in rows:
        with np.load(row["path"], allow_pickle=False) as obj:
            reader_counts.append(len(obj["masks"]))
    if any(n < 3 for n in reader_counts):
        raise ValueError("Each case needs visible, revealable and hidden readers")
    rotations = math.lcm(*set(reader_counts))
    for hidden_rotation in range(rotations):
        for seed in args.seeds:
            seed_all(seed)
            rng = np.random.default_rng(seed)
            cases = []
            with torch.inference_mode():
                for row in rows:
                    with np.load(row["path"], allow_pickle=False) as obj:
                        number = len(obj["masks"])
                    hidden = hidden_rotation % number
                    order = rng.permutation(
                        [j for j in range(number) if j != hidden]
                    ).tolist()
                    initial = order.pop(0)
                    image, masks = field(row, initial)
                    x = image_tensor(image, args.device)
                    features = model.encode_image(x)
                    case = {
                        "row": row,
                        "image": x,
                        "masks": masks,
                        "features": features,
                        "visible": [initial],
                        "pool": order,
                        "hidden": hidden,
                        "embeddings": model.embed_masks(
                            x,
                            torch.as_tensor(
                                masks[initial : initial + 1], device=args.device
                            ),
                        ),
                        "normal": torch.randn(cfg["samples"], 8, device=args.device),
                        "mixture_draws": torch.rand(cfg["samples"]).numpy(),
                        "uniforms": torch.rand(
                            cfg["samples"], *image.shape, device=args.device
                        ),
                    }
                    cases.append(case)
            if not cases:
                continue
            patients = defaultdict(int)
            tasks = defaultdict(set)
            for case in cases:
                row = case["row"]
                task = row.get("task", row["dataset"])
                patients[(task, row["patient_id"])] += 1
                tasks[task].add(row["patient_id"])
            for case in cases:
                row = case["row"]
                task = row.get("task", row["dataset"])
                case["weight"] = 1 / (
                    len(tasks) * len(tasks[task]) * patients[(task, row["patient_id"])]
                )
            total = sum(len(c["pool"]) for c in cases)
            checkpoints = defaultdict(list)
            for fraction in cfg["budgets"]:
                checkpoints[int(np.floor(fraction * total))].append(fraction)
            heap, cursor = [], 0

            def prediction(case):
                state = model.statistics(case["embeddings"])
                samples, marginal = model.sample(
                    case["features"], state, case["normal"], case["uniforms"]
                )
                if args.policy == "empirical":
                    visible = case["masks"][case["visible"]]
                    prior_samples, prior_mean = samples, marginal
                    chosen = np.floor(
                        case["mixture_draws"] * (len(visible) + 1)
                    ).astype(int)
                    samples = np.stack(
                        [
                            prior_samples[k] if j == len(visible) else visible[j]
                            for k, j in enumerate(chosen)
                        ]
                    )
                    marginal = (visible.sum(0) + prior_mean) / (len(visible) + 1)
                return state, samples, marginal

            def priority(case):
                if args.policy in (
                    "setgain",
                    "no_count",
                    "no_disagreement",
                    "dice_target",
                    "shuffled",
                    "no_exploration",
                    "no_set",
                    "current_loss",
                    "feature_lal",
                ):
                    state = model.statistics(case["embeddings"])
                    vector = model.gain_features(
                        case["features"],
                        state,
                        torch.as_tensor(
                            case["masks"][case["visible"]], device=args.device
                        ),
                        len(case["pool"]),
                        args.policy,
                    )
                    estimated = (
                        lal.predict(vector.cpu().numpy()[None, -13:])[0]
                        if args.policy == "feature_lal"
                        else head(vector).item()
                    )
                    return float(estimated) * case["weight"]
                state, samples, marginal = prediction(case)
                vector = model.gain_features(
                    case["features"],
                    state,
                    torch.as_tensor(case["masks"][case["visible"]], device=args.device),
                    len(case["pool"]),
                    args.policy,
                )
                entropy = float(
                    (
                        -(
                            marginal * np.log(marginal + 1e-6)
                            + (1 - marginal) * np.log(1 - marginal + 1e-6)
                        )
                    ).mean()
                )
                if args.policy == "disagreement":
                    return (
                        pair_distance(case["masks"][case["visible"]])
                        if len(case["visible"]) > 1
                        else entropy
                    )
                if args.policy == "entropy":
                    return entropy
                if args.policy == "diversity_entropy":
                    feature = case["features"][-1].mean((-2, -1))[0].cpu().numpy()
                    others = [
                        c["features"][-1].mean((-2, -1))[0].cpu().numpy()
                        for c in cases
                        if len(c["visible"]) > 1
                    ]
                    diversity = (
                        min(np.linalg.norm(feature - f) for f in others)
                        if others
                        else 1.0
                    )
                    return entropy * diversity
                if args.policy in ("bald", "epig"):
                    probabilities = []
                    for auxiliary in ensemble:
                        fs = auxiliary.encode_image(case["image"])
                        ss = auxiliary.statistics(
                            auxiliary.embed_masks(
                                case["image"],
                                torch.as_tensor(
                                    case["masks"][case["visible"]], device=args.device
                                ),
                            )
                        )
                        probabilities.append(
                            auxiliary.probabilities(fs, ss, case["normal"])
                            .mean(0)
                            .cpu()
                            .numpy()
                        )
                    p = np.stack(probabilities).clip(1e-6, 1 - 1e-6)
                    mean = p.mean(0)
                    h = lambda a: -(a * np.log(a) + (1 - a) * np.log(1 - a))
                    if args.policy == "bald":
                        return float((h(mean) - h(p).mean(0)).mean())
                    joint = np.stack(
                        [
                            (p * p).mean(0),
                            (p * (1 - p)).mean(0),
                            (p * (1 - p)).mean(0),
                            ((1 - p) ** 2).mean(0),
                        ]
                    )
                    independent = np.stack(
                        [mean**2, mean * (1 - mean), mean * (1 - mean), (1 - mean) ** 2]
                    )
                    return float(
                        (joint * np.log((joint + 1e-6) / (independent + 1e-6)))
                        .sum(0)
                        .mean()
                    )
                if args.policy == "oracle":
                    target = case["masks"][case["hidden"] : case["hidden"] + 1]
                    before = energy_score(samples, target)
                    values = []
                    for candidate in case["pool"]:
                        emb = model.embed_masks(
                            case["image"],
                            torch.as_tensor(
                                case["masks"][candidate : candidate + 1],
                                device=args.device,
                            ),
                        )
                        after_state = model.statistics(
                            torch.cat([case["embeddings"], emb])
                        )
                        after, _ = model.sample(
                            case["features"],
                            after_state,
                            case["normal"],
                            case["uniforms"],
                        )
                        values.append(before - energy_score(after, target))
                    return float(np.mean(values)) * case["weight"]
                return 0.0

            with torch.inference_mode():
                start = time.perf_counter()
                for index, case in enumerate(cases):
                    case["version"] = 0
                    heapq.heappush(heap, (-priority(case), index, 0))
                sync()
                score_ms = 1000 * (time.perf_counter() - start)
                states = len(cases)
                if args.score_states:
                    sync()
                    start = time.perf_counter()
                    for index in range(args.score_states):
                        priority(cases[index % len(cases)])
                    sync()
                    score_ms = 1000 * (time.perf_counter() - start)
                    states = args.score_states
                timing.append(
                    {"states": states, "unique_cases": len(cases), "score_ms": score_ms}
                )
                for purchased in range(total + 1):
                    if purchased in checkpoints:
                        evaluated = []
                        for case in cases:
                            _, samples, marginal = prediction(case)
                            evaluated.append(
                                dict(
                                    case["row"],
                                    metrics=predictive_metrics(
                                        samples,
                                        marginal,
                                        case["masks"],
                                        case["hidden"],
                                        case["masks"][case["visible"][0]],
                                    ),
                                )
                            )
                        for fraction in checkpoints[purchased]:
                            repeats.append(
                                {
                                    "method": args.policy,
                                    "budget": fraction,
                                    "seed": seed,
                                    "hidden_rotation": hidden_rotation,
                                    "metrics": weighted_mean(evaluated),
                                    "cases": evaluated,
                                }
                            )
                    if purchased == total:
                        break
                    eligible = [j for j, c in enumerate(cases) if c["pool"]]
                    if args.policy in ("random", "empirical") or (
                        args.policy
                        in (
                            "setgain",
                            "no_count",
                            "no_disagreement",
                            "dice_target",
                            "shuffled",
                            "no_set",
                        )
                        and rng.random() < cfg["exploration"]
                    ):
                        index = int(rng.choice(eligible))
                    elif args.policy == "uniform":
                        index = min(eligible, key=lambda j: (j - cursor) % len(cases))
                        cursor = (index + 1) % len(cases)
                    elif args.policy == "fewest":
                        index = min(
                            eligible,
                            key=lambda j: (
                                len(cases[j]["visible"]),
                                (j - cursor) % len(cases),
                            ),
                        )
                        cursor = (index + 1) % len(cases)
                    else:
                        if args.policy == "diversity_entropy":
                            heap = [
                                (-priority(cases[j]), j, cases[j]["version"])
                                for j in eligible
                            ]
                            heapq.heapify(heap)
                        while heap:
                            _, index, version = heapq.heappop(heap)
                            if (
                                cases[index]["pool"]
                                and version == cases[index]["version"]
                            ):
                                break
                        else:
                            raise RuntimeError("No eligible acquisition heap entry")
                    case = cases[index]
                    state, before, _ = prediction(case)
                    gain_vector = model.gain_features(
                        case["features"],
                        state,
                        torch.as_tensor(
                            case["masks"][case["visible"]], device=args.device
                        ),
                        len(case["pool"]),
                        args.policy,
                    )
                    predicted = (
                        float(lal.predict(gain_vector.cpu().numpy()[None, -13:])[0])
                        if lal is not None
                        else float(head(gain_vector).item())
                        if args.head and args.policy != "current_loss"
                        else None
                    )
                    candidate = case["pool"].pop(0)
                    start = time.perf_counter()
                    case["visible"].append(candidate)
                    emb = model.embed_masks(
                        case["image"],
                        torch.as_tensor(
                            case["masks"][candidate : candidate + 1], device=args.device
                        ),
                    )
                    case["embeddings"] = torch.cat([case["embeddings"], emb])
                    sync()
                    update_ms = 1000 * (time.perf_counter() - start)
                    start_emission = time.perf_counter()
                    _, after, _ = prediction(case)
                    sync()
                    emission_ms = 1000 * (time.perf_counter() - start_emission)
                    elapsed = update_ms + emission_ms
                    target = case["masks"][case["hidden"] : case["hidden"] + 1]
                    gain = energy_score(before, target) - energy_score(after, target)
                    query_rows.append(
                        {
                            "id": case["row"]["id"],
                            "seed": seed,
                            "hidden_rotation": hidden_rotation,
                            "predicted_gain": predicted,
                            "realized_gain": gain,
                            "update_ms": update_ms,
                            "emission_ms": emission_ms,
                            "update_and_emission_ms": elapsed,
                        }
                    )
                    case["version"] += 1
                    if case["pool"]:
                        heapq.heappush(heap, (-priority(case), index, case["version"]))
    summary = []
    for budget in cfg["budgets"]:
        case_metrics = [
            case
            for repeat in repeats
            if repeat["budget"] == budget
            for case in repeat["cases"]
        ]
        values = [weighted_mean(case_metrics)] if case_metrics else []
        if values:
            summary.append(
                {
                    "method": args.policy,
                    "budget": budget,
                    **{
                        k: float(np.mean([r[k] for r in values if r[k] is not None]))
                        if any(r[k] is not None for r in values)
                        else None
                        for k in values[0]
                    },
                }
            )
    if not summary:
        raise ValueError("No eligible cases for evaluation")
    budget_width = summary[-1]["budget"] - summary[0]["budget"]
    if budget_width <= 0:
        raise ValueError("AURC needs at least two increasing budget fractions")
    aurc = float(
        trapezoid([r["energy_score"] for r in summary], [r["budget"] for r in summary])
        / budget_width
    )
    paired = [r for r in query_rows if r["predicted_gain"] is not None]
    mechanism = {
        "aurc": aurc,
        "negative_gain_fraction": float(
            np.mean([r["realized_gain"] < 0 for r in query_rows])
        )
        if query_rows
        else None,
        "gain_mae": float(
            np.mean([abs(r["predicted_gain"] - r["realized_gain"]) for r in paired])
        )
        if paired
        else None,
        "gain_spearman": float(
            spearmanr(
                [r["predicted_gain"] for r in paired],
                [r["realized_gain"] for r in paired],
            ).statistic
        )
        if len(paired) > 1
        else None,
    }
    bins = []
    if paired:
        ordered = sorted(paired, key=lambda r: r["predicted_gain"])
        for subset in np.array_split(np.arange(len(ordered)), min(10, len(ordered))):
            bins.append(
                {
                    "method": args.policy,
                    "predicted_gain": float(
                        np.mean([ordered[i]["predicted_gain"] for i in subset])
                    ),
                    "realized_gain": float(
                        np.mean([ordered[i]["realized_gain"] for i in subset])
                    ),
                }
            )
    mechanism.update(peak_memory_mib())
    mechanism["reader_rotations"] = rotations
    mechanism["score_states"] = float(np.mean([r["states"] for r in timing]))
    mechanism["score_ms"] = float(np.mean([r["score_ms"] for r in timing]))
    for key in ("update_ms", "emission_ms"):
        mechanism[key] = (
            float(np.mean([r[key] for r in query_rows])) if query_rows else None
        )
    mechanism["total_ms"] = (
        (mechanism["score_ms"] + mechanism["update_ms"] + mechanism["emission_ms"])
        if query_rows
        else None
    )
    save(
        Path(args.output) / "results.json",
        {
            "summary": summary,
            "mechanism": mechanism,
            "repeats": repeats,
            "queries": query_rows,
            "calibration": bins,
            "timing": timing,
            "config": cfg,
        },
    )
    save(Path(args.output) / "calibration.json", {"summary": bins})


def main():
    parser = argparse.ArgumentParser(prog="setgain")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source", choices=["lidc", "qubiq", "manifest"], required=True)
    prep.add_argument("--input")
    prep.add_argument("--output", required=True)
    predictor = commands.add_parser("train-predictor")
    predictor.add_argument("--manifest", required=True)
    predictor.add_argument("--output", required=True)
    predictor.add_argument("--no-set", action="store_true")
    targets = commands.add_parser("make-targets")
    targets.add_argument("--manifest", required=True)
    targets.add_argument("--checkpoint", required=True)
    targets.add_argument("--output", required=True)
    targets.add_argument(
        "--variant",
        choices=["setgain", "no_count", "no_disagreement", "dice_target", "shuffled"],
        default="setgain",
    )
    gain = commands.add_parser("train-gain")
    gain.add_argument("--targets", required=True)
    gain.add_argument("--output", required=True)
    gain.add_argument(
        "--variant",
        choices=[
            "setgain",
            "no_count",
            "no_disagreement",
            "dice_target",
            "shuffled",
            "current_loss",
        ],
        default="setgain",
    )
    execution = commands.add_parser("run")
    execution.add_argument("--manifest", required=True)
    execution.add_argument("--checkpoint", required=True)
    execution.add_argument("--head")
    execution.add_argument("--targets")
    execution.add_argument("--ensemble", nargs="*", default=[])
    execution.add_argument("--policy", choices=POLICIES, default="setgain")
    execution.add_argument("--split", choices=["validation", "test"], default="test")
    execution.add_argument(
        "--seeds", type=int, nargs="+", default=list(range(100, 110))
    )
    execution.add_argument("--score-states", type=int)
    execution.add_argument("--output", required=True)
    for sub in (predictor, targets, gain, execution):
        sub.add_argument("--config", default="configs/default.json")
        sub.add_argument("--seed", type=int, default=17)
        sub.add_argument(
            "--device", default="cuda" if torch.cuda.is_available() else "cpu"
        )
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--inputs", nargs="+", required=True)
    evaluate.add_argument("--output", required=True)
    plot = commands.add_parser("plot")
    plot.add_argument("--input", required=True)
    plot.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.source == "lidc":
            prepare_lidc(args.output, readers=4)
        elif args.source == "qubiq":
            from .qubiq import convert

            convert(args.input, args.output)
        else:
            prepare_manifest(args.input, args.output)
    elif args.command == "train-predictor":
        train_predictor(args)
    elif args.command == "make-targets":
        make_targets(args)
    elif args.command == "train-gain":
        train_gain(args)
    elif args.command == "run":
        run(args)
    elif args.command == "evaluate":
        results = [load(p) for p in args.inputs]
        save(
            args.output,
            {
                "summary": [
                    dict(row, **obj["mechanism"])
                    for obj in results
                    for row in obj["summary"]
                ]
            },
        )
    else:
        plot_results(args.input, args.output)


if __name__ == "__main__":
    main()
