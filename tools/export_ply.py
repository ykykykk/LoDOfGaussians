"""Export a finest-leaf Gaussian PLY without superimposing ancestor LoDs."""
import argparse
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def finest_leaf_indices(nodes: torch.Tensor, *, include_skybox: bool = False) -> torch.Tensor:
    if nodes.ndim != 2 or nodes.shape[1] != 6 or nodes.dtype not in (torch.int32, torch.int64):
        raise ValueError("Expected integer hierarchy nodes [N, 6]")
    # Leaves can have different depths in an adaptive tree. Do not use max(depth).
    keep = nodes[:, 2] == 0
    if include_skybox:
        # ALoD stores independent skybox rows with -99 in these fields.
        keep = keep | ((nodes[:, 0] == -99) & (nodes[:, 1] == -99) & (nodes[:, 2] == -99))
    return torch.where(keep)[0].cpu()


def export_gaussian_tensors_ply(
    xyz, dc, rest, opacity, scale, rotation, output: Path,
    chunk_size: int = 262_144, opacity_activated: bool = False,
    *, indices: Optional[torch.Tensor] = None,
) -> None:
    tensors = {name: value.detach().cpu() for name, value in (
        ("xyz", xyz), ("dc", dc), ("rest", rest), ("opacity", opacity),
        ("scale", scale), ("rotation", rotation),
    )}
    count = tensors["xyz"].shape[0]
    if chunk_size <= 0 or any(value.shape[0] != count for value in tensors.values()):
        raise ValueError("Invalid chunk size or inconsistent Gaussian tensor lengths")
    for name, width in (("xyz", 3), ("dc", 3), ("opacity", 1), ("scale", 3), ("rotation", 4)):
        if tensors[name].shape != (count, width):
            raise ValueError(f"Expected {name} [N, {width}]")
    if tensors["rest"].ndim not in (2, 3):
        raise ValueError("Expected SH-rest [N, width] or [N, coefficients, 3]")
    if tensors["rest"].ndim == 3 and tensors["rest"].shape[2] != 3:
        raise ValueError("SH-rest's last dimension must be RGB")
    rest_width = int(np.prod(tensors["rest"].shape[1:]))
    if rest_width not in (0, 9, 24, 45):
        raise ValueError("Only SH degrees 0, 1, 2 and 3 are supported")
    if indices is not None:
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected one-dimensional integer export indices")
        indices = indices.detach().to(device="cpu", dtype=torch.long)
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= count):
            raise ValueError("Export index outside the Gaussian tensor range")
    output_count = count if indices is None else indices.numel()
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(rest_width)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    dtype = np.dtype([(name, "<f4") for name in names])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=output.parent, prefix=output.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            header = ["ply", "format binary_little_endian 1.0", f"element vertex {output_count}"]
            header += [f"property float {name}" for name in names]
            handle.write(("\n".join([*header, "end_header"]) + "\n").encode("ascii"))
            for start in range(0, output_count, chunk_size):
                stop = min(start + chunk_size, output_count)
                selected = slice(start, stop) if indices is None else indices[start:stop]
                values = {name: value[selected] for name, value in tensors.items()}
                for name, value in values.items():
                    if not torch.isfinite(value).all():
                        raise ValueError(f"Non-finite {name} in export rows {start}:{stop}")
                chunk = np.zeros(stop - start, dtype=dtype)
                for i, name in enumerate(("x", "y", "z")):
                    chunk[name] = values["xyz"][:, i].numpy()
                for i in range(3):
                    chunk[f"f_dc_{i}"] = values["dc"][:, i].numpy()
                sh = values["rest"].reshape(stop - start, rest_width // 3, 3)
                sh = sh.permute(0, 2, 1).reshape(stop - start, rest_width).numpy()
                for i in range(rest_width):
                    chunk[f"f_rest_{i}"] = sh[:, i]
                alpha = values["opacity"].reshape(-1)
                chunk["opacity"] = (torch.logit(alpha.clamp(1e-6, 1 - 1e-6)) if opacity_activated else alpha).numpy()
                # .dhier already stores log-scales and wxyz rotations; do not log twice.
                for name, prefix in (("scale", "scale"), ("rotation", "rot")):
                    array = values[name].numpy()
                    for i in range(array.shape[1]):
                        chunk[f"{prefix}_{i}"] = array[:, i]
                chunk.tofile(handle)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_hierarchy_ply(source: Path, output: Path, chunk_size: int = 262_144, *, include_skybox: bool = False) -> None:
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Input hierarchy and output PLY must be different files")
    if not source.is_file():
        raise FileNotFoundError(source)
    # Lazy import: tensor export and its CPU tests do not require CUDA extensions.
    from gaussian_hierarchy._C import load_dynamic_hierarchy
    xyz, sh, opacity, scale, rotation, nodes = load_dynamic_hierarchy(str(source))
    if nodes.shape[0] != xyz.shape[0]:
        raise ValueError("Hierarchy node and Gaussian counts differ")
    selected = finest_leaf_indices(nodes, include_skybox=include_skybox)
    leaves = int((nodes[:, 2] == 0).sum())
    internal = int((nodes[:, 2] > 0).sum())
    print(f"Hierarchy nodes: {len(xyz):,}; finest leaves: {leaves:,}; internal LoD nodes excluded: {internal:,}; PLY vertices: {len(selected):,}")
    export_gaussian_tensors_ply(
        xyz, sh[:, 0], sh[:, 1:], opacity, scale, rotation, output,
        chunk_size, opacity_activated=True, indices=selected,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the finest leaf cut of an ALoD .dhier file to binary Gaussian PLY")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=262_144)
    parser.add_argument("--include-skybox", action="store_true", help="Also retain independent background skybox rows (never ancestor LoDs)")
    args = parser.parse_args()
    export_hierarchy_ply(args.input, args.output, args.chunk_size, include_skybox=args.include_skybox)


if __name__ == "__main__":
    main()
