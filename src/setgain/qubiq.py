from pathlib import Path

import nibabel as nib
import numpy as np

from .utils import load, save, split_groups


def convert(mapping, output):
    source, output = Path(mapping), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for row in load(source):
        if row["task"] not in ("brain-growth", "prostate", "kidney"):
            raise ValueError("Primary QUBIQ tasks must have individual human contours")
        image_file = nib.load(source.parent / row["image"])
        image = image_file.get_fdata(dtype=np.float32)
        masks = []
        for label_path in row["labels"]:
            label_file = nib.load(source.parent / label_path)
            if label_file.shape != image_file.shape or not np.allclose(
                label_file.affine, image_file.affine, atol=1e-4
            ):
                raise ValueError("QUBIQ image/label geometry differs")
            values = label_file.get_fdata()
            masks.append(values == row.get("structure_label", 1))
        if len(masks) < 3:
            raise ValueError("Need at least three independent human reader masks")
        if image.ndim == 3:
            if "slice" not in row:
                raise ValueError(
                    "Specify a curator-selected image slice before reader acquisition"
                )
            image = image[:, :, row["slice"]]
            masks = [m[:, :, row["slice"]] for m in masks]
        if image.ndim != 2:
            raise ValueError("QUBIQ exports must define a two-dimensional image field")
        path = output / (row["id"] + ".npz")
        np.savez_compressed(
            path,
            image=image,
            masks=np.asarray(masks, dtype=np.uint8),
            spacing=np.asarray(image_file.header.get_zooms()[:2]),
        )
        records.append(
            {
                "id": row["id"],
                "patient_id": row["patient_id"],
                "dataset": "QUBIQ",
                "task": row["task"],
                "modality": row.get("modality", "MRI"),
                "path": path.name,
            }
        )
    if not records:
        raise ValueError("Empty QUBIQ source mapping")
    assigned = []
    for task in sorted({r["task"] for r in records}):
        assigned.extend(
            split_groups([r for r in records if r["task"] == task], "patient_id")
        )
    save(output / "manifest.json", assigned)
