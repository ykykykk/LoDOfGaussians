import argparse
from pathlib import Path

import numpy as np
import torch
from gaussian_hierarchy._C import load_dynamic_hierarchy


def export_gaussian_tensors_ply(
    xyz,
    dc,
    rest,
    opacity,
    scale,
    rotation,
    output: Path,
    chunk_size: int = 262_144,
    opacity_activated: bool = False,
) -> None:
    tensors = {
        "xyz": xyz.detach().cpu(),
        "dc": dc.detach().cpu(),
        "rest": rest.detach().cpu(),
        "opacity": opacity.detach().cpu(),
        "scale": scale.detach().cpu(),
        "rotation": rotation.detach().cpu(),
    }
    count = tensors["xyz"].shape[0]
    if chunk_size <= 0 or any(value.shape[0] != count for value in tensors.values()):
        raise ValueError("invalid chunk size or inconsistent hierarchy tensor lengths")
    if tensors["xyz"].shape != (count, 3) or tensors["opacity"].shape != (count, 1):
        raise ValueError("expected positions [N, 3] and opacity [N, 1]")
    if tensors["scale"].shape != (count, 3) or tensors["rotation"].shape != (count, 4):
        raise ValueError("expected scales [N, 3] and rotations [N, 4]")
    if tensors["dc"].shape != (count, 3) or tensors["rest"].ndim not in (2, 3):
        raise ValueError("expected DC [N, 3] and rest SH [N, width] or [N, coefficients, 3]")

    rest_width = tensors["rest"].shape[1] if tensors["rest"].ndim == 2 else tensors["rest"].shape[1] * 3
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(rest_width)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    dtype = np.dtype([(name, "<f4") for name in names])
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("wb") as handle:
        header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
        header += [f"property float {name}" for name in names]
        handle.write(("\n".join([*header, "end_header"]) + "\n").encode("ascii"))
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            for name, value in tensors.items():
                if not torch.isfinite(value[start:stop]).all():
                    raise ValueError(f"non-finite {name} values in rows {start}:{stop}")
            chunk = np.empty(stop - start, dtype=dtype)
            positions = tensors["xyz"][start:stop].numpy()
            for index, name in enumerate(("x", "y", "z")):
                chunk[name] = positions[:, index]
            chunk["nx"], chunk["ny"], chunk["nz"] = 0, 0, 0
            dc = tensors["dc"][start:stop].numpy()
            for index in range(3):
                chunk[f"f_dc_{index}"] = dc[:, index]
            rest = tensors["rest"][start:stop].reshape(stop - start, rest_width // 3, 3)
            rest = rest.permute(0, 2, 1).reshape(stop - start, rest_width).numpy()
            for index in range(rest_width):
                chunk[f"f_rest_{index}"] = rest[:, index]
            opacity = tensors["opacity"][start:stop].reshape(-1)
            chunk["opacity"] = torch.logit(opacity.clamp(1e-6, 1 - 1e-6)).numpy() if opacity_activated else opacity.numpy()
            for values, prefix in ((tensors["scale"][start:stop].numpy(), "scale"), (tensors["rotation"][start:stop].numpy(), "rot")):
                for index in range(values.shape[1]):
                    chunk[f"{prefix}_{index}"] = values[:, index]
            chunk.tofile(handle)


def export_hierarchy_ply(source: Path, output: Path, chunk_size: int = 262_144) -> None:
    xyz, sh, opacity, scale, rotation, _ = load_dynamic_hierarchy(str(source))
    export_gaussian_tensors_ply(
        xyz,
        sh[:, 0],
        sh[:, 1:],
        opacity,
        scale,
        rotation,
        output,
        chunk_size,
        opacity_activated=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export every Gaussian in an ALoD .dhier file to binary PLY")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=262_144)
    args = parser.parse_args()
    export_hierarchy_ply(args.input, args.output, args.chunk_size)


if __name__ == "__main__":
    main()
