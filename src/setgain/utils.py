import json
import random
from pathlib import Path

import numpy as np


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple, np.ndarray)):
            return [clean(v) for v in item]
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.bool_):
            return bool(item)
        return item

    path.write_text(
        json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def seed_all(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_groups(rows, key, seed=2027):
    groups = sorted({str(r[key]) for r in rows})
    order = np.random.default_rng(seed).permutation(groups)
    first = int(0.6 * len(order))
    second = first + int(0.2 * len(order))
    assigned = {
        g: "train" if i < first else "validation" if i < second else "test"
        for i, g in enumerate(order)
    }
    return [dict(r, split=assigned[str(r[key])]) for r in rows]


def plot_results(path, output, metrics=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obj = load(path)
    rows = obj["summary"] if isinstance(obj, dict) else obj
    if isinstance(rows, dict):
        rows = [dict(method=k, **v) for k, v in rows.items()]
    if not rows:
        raise ValueError("No result rows to plot")
    rows = [
        dict(
            row,
            **{
                key + "_" + child: value
                for key, obj in row.items()
                if isinstance(obj, dict)
                for child, value in obj.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            },
        )
        for row in rows
    ]
    metrics = metrics or [
        k
        for k, v in rows[0].items()
        if isinstance(v, (int, float)) and k not in {"seed", "budget"}
    ]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        valid = [r for r in rows if isinstance(r.get(metric), (int, float))]
        if not valid:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        x_key = (
            "budget"
            if all("budget" in r for r in valid)
            else "opportunity_density"
            if all("opportunity_density" in r for r in valid)
            else "top_k"
            if all("top_k" in r for r in valid)
            else "predicted_gain"
            if metric == "realized_gain" and all("predicted_gain" in r for r in valid)
            else None
        )
        if x_key is not None:
            for method in sorted({r["method"] for r in valid}):
                subset = sorted(
                    (r for r in valid if r["method"] == method), key=lambda r: r[x_key]
                )
                ax.plot(
                    [r[x_key] for r in subset],
                    [r[metric] for r in subset],
                    "o-",
                    label=method,
                )
            ax.set_xlabel(x_key.replace("_", " ").title())
            ax.legend(fontsize=8)
        else:
            ax.barh(
                [r.get("method", str(i)) for i, r in enumerate(valid)],
                [r[metric] for r in valid],
                color="#487d96",
            )
        ax.set_ylabel(metric)
        fig.tight_layout()
        fig.savefig(output / (metric + ".png"), dpi=180)
        fig.savefig(output / (metric + ".svg"))
        plt.close(fig)


def peak_memory_mib():
    import os
    import sys
    import torch

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak_working",
                    "working",
                    "peak_paged",
                    "paged",
                    "peak_nonpaged",
                    "nonpaged",
                    "pagefile",
                    "peak_pagefile",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        api = ctypes.windll.psapi.GetProcessMemoryInfo
        api.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not api(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        process = counters.peak_working / 1024**2
    else:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        process = peak / (1024**2 if sys.platform == "darwin" else 1024)
    gpu = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else None
    )
    return {"peak_process_memory_mib": float(process), "peak_gpu_memory_mib": gpu}
