from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from .utils import load, save, split_groups


def resize(array, shape, order=1):
    return ndi.zoom(
        array, np.asarray(shape) / np.asarray(array.shape), order=order, prefilter=False
    )


def normalize(image, modality):
    image = np.asarray(image, dtype=np.float32)
    if modality == "CT":
        return (np.clip(image, -1000, 1000) + 1000) / 2000
    values = image[image != 0]
    if not len(values):
        values = image.reshape(-1)
    return (image - values.mean()) / max(float(values.std()), 1e-6)


def crop(array, center, shape, fill=0):
    shape = np.asarray(shape, dtype=int)
    start = np.asarray(center, dtype=int) - shape // 2
    end = start + shape
    output = np.full(tuple(shape), fill, dtype=array.dtype)
    low, high = np.maximum(start, 0), np.minimum(end, array.shape)
    if np.any(high <= low):
        return output
    output[tuple(slice(int(a), int(b)) for a, b in zip(low - start, high - start))] = (
        array[tuple(slice(int(a), int(b)) for a, b in zip(low, high))]
    )
    return output


def prepare_acdc(raw, output, seed=2027):
    raw, output = Path(raw), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for mask_path in sorted(raw.rglob("patient*_frame*_gt.nii.gz")):
        image_path = mask_path.with_name(mask_path.name.replace("_gt", ""))
        mask_file, image_file = nib.load(mask_path), nib.load(image_path)
        image = normalize(image_file.get_fdata(), "MRI")
        labels = mask_file.get_fdata().astype(np.uint8)
        spacing = image_file.header.get_zooms()[:2]
        patient = mask_path.parent.name
        group = "unknown"
        info = mask_path.parent / "Info.cfg"
        if info.exists():
            group = next(
                (
                    r.split(":", 1)[1].strip()
                    for r in info.read_text().splitlines()
                    if r.startswith("Group:")
                ),
                "unknown",
            )
        for structure in (1, 2, 3):
            for z in range(image.shape[2]):
                target = labels[:, :, z] == structure
                if not target.any():
                    continue
                identifier = f"{mask_path.name.split('.')[0]}_s{structure}_z{z}"
                path = output / (identifier + ".npz")
                shape = (256, 256)
                np.savez_compressed(
                    path,
                    image=resize(image[:, :, z], shape).astype(np.float32),
                    masks=resize(target.astype(np.uint8), shape, 0)[None],
                    spacing=np.asarray(spacing) * np.asarray(target.shape) / shape,
                )
                rows.append(
                    {
                        "id": identifier,
                        "patient_id": patient,
                        "dataset": "ACDC",
                        "group": group,
                        "path": path.name,
                    }
                )
    assigned = []
    for group in sorted({r["group"] for r in rows}):
        assigned.extend(
            split_groups([r for r in rows if r["group"] == group], "patient_id", seed)
        )
    if not assigned:
        raise ValueError("No labeled ACDC volumes found")
    save(output / "manifest.json", assigned)


def prepare_lidc(output, readers=1, seed=2027):
    import pylidc as pl

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for scan in pl.query(pl.Scan).all():
        volume = scan.to_volume().astype(np.float32)
        for index, annotations in enumerate(scan.cluster_annotations()):
            if (
                len(annotations) < readers
                or (readers == 4 and len(annotations) != 4)
                or np.median([a.diameter for a in annotations]) < 3
            ):
                continue
            masks = []
            for annotation in annotations:
                mask = np.zeros(volume.shape, dtype=np.uint8)
                mask[annotation.bbox()] = annotation.boolean_mask().astype(np.uint8)
                masks.append(mask)
            identifier = f"{scan.patient_id}_scan{scan.id}_n{index}"
            path = output / (identifier + ".npz")
            np.savez_compressed(
                path,
                image=volume,
                masks=np.stack(masks),
                spacing=np.asarray(
                    [scan.pixel_spacing, scan.pixel_spacing, scan.slice_thickness]
                ),
            )
            rows.append(
                {
                    "id": identifier,
                    "patient_id": scan.patient_id,
                    "dataset": "LIDC",
                    "modality": "CT",
                    "path": path.name,
                }
            )
    if not rows:
        raise ValueError("No eligible LIDC nodules found; configure pylidc DICOM paths")
    save(output / "manifest.json", split_groups(rows, "patient_id", seed))


def prepare_manifest(source, output, seed=2027):
    source, output = Path(source), Path(output)
    rows = load(source)
    for row in rows:
        path = (source.parent / row["path"]).resolve()
        with np.load(path, allow_pickle=False) as case:
            if "image" not in case or "masks" not in case or "spacing" not in case:
                raise ValueError(f"Missing image, masks, or spacing: {path}")
            if case["masks"].shape[1:] != case["image"].shape:
                raise ValueError(f"Image and masks have different shapes: {path}")
            if not np.isin(case["masks"], [0, 1]).all():
                raise ValueError(f"Masks must be binary: {path}")
        row["path"] = str(path)
    save(output, split_groups(rows, "patient_id", seed))


def records(manifest, split):
    path = Path(manifest)
    rows = [r for r in load(path) if r["split"] == split]
    for row in rows:
        row["path"] = str((path.parent / row["path"]).resolve())
    if not rows:
        raise ValueError(f"No cases in {split} split")
    return rows


def fixed_field(row, initial=0, size=256, side_mm=96):
    with np.load(row["path"], allow_pickle=False) as obj:
        image = obj["image"].astype(np.float32)
        masks = obj["masks"].astype(np.uint8)
        spacing = obj["spacing"].astype(float)
    if image.ndim == 3:
        reference = masks[initial]
        z = int(reference.sum((0, 1)).argmax())
        image, masks = image[:, :, z], masks[:, :, :, z]
        points = np.argwhere(masks[initial])
        if not len(points):
            raise ValueError("Initial contour is empty")
        center = np.rint(points.mean(0)).astype(int)
        shape = np.maximum(1, np.rint(side_mm / spacing[:2]).astype(int))
        image = crop(image, center, shape, -1000)
        masks = np.stack([crop(mask, center, shape) for mask in masks])
    old_shape = image.shape
    image = (
        normalize(
            image, row.get("modality", "CT" if row["dataset"] == "LIDC" else "MRI")
        )
        if row["dataset"] != "ACDC"
        else image
    )
    image = resize(image, (size, size)).astype(np.float32)
    masks = np.stack([resize(mask, (size, size), 0) for mask in masks]).astype(
        np.float32
    )
    spacing = spacing[:2] * np.asarray(old_shape) / size
    return image, masks, spacing
