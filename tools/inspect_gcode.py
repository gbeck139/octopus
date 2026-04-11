"""Quantitative diagnostics for 4-axis (XZCB) G-code.

Emits a JSON report and a short human-readable summary. Designed as a
text-first "vision" channel: most slicer bugs (missing layers, runaway
travels, bad kinematics, singularity hits) show up in numbers before they
show up in pictures.

Usage:
    python inspect_gcode.py --gcode path/to/file.gcode
    python inspect_gcode.py --gcode new.gcode --baseline old_stats.json
    python inspect_gcode.py --gcode f.gcode --out tools/vision_cache/foo/latest
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np


NOZZLE_OFFSET = 43.0
LONG_TRAVEL_MM = 50.0
Z_JUMP_MM = 5.0
B_SINGULARITY_DEG = 5.0


def machine_to_cartesian(x, z, c, b, nozzle_offset=NOZZLE_OFFSET):
    rotation_rad = np.deg2rad(-b)
    r = x - np.sin(rotation_rad) * nozzle_offset
    z_model = z - (np.cos(rotation_rad) - 1) * nozzle_offset
    c_rad = np.deg2rad(c)
    return r * np.cos(c_rad), r * np.sin(c_rad), z_model


def parse_gcode(path):
    pos = {"X": 0.0, "Z": 20.0, "C": 0.0, "B": -15.0}
    moves = []
    line_re = re.compile(r"([XZCBE])([-\d\.]+)")

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            if not (line.startswith("G0") or line.startswith("G1")):
                continue
            matches = line_re.findall(line)
            if not matches:
                continue

            extruded = 0.0
            changed = False
            for tag, val_str in matches:
                val = float(val_str)
                if tag == "E":
                    if val > 0:
                        extruded = val
                else:
                    pos[tag] = val
                    changed = True

            if changed:
                moves.append((pos["X"], pos["Z"], pos["C"], pos["B"], extruded))

    return np.array(moves, dtype=float) if moves else np.zeros((0, 5))


def compute_stats(moves):
    n = len(moves)
    if n == 0:
        return {"empty": True, "num_moves": 0}

    X, Z, C, B, E = moves[:, 0], moves[:, 1], moves[:, 2], moves[:, 3], moves[:, 4]
    is_ext = E > 0

    cart = np.array([machine_to_cartesian(X[i], Z[i], C[i], B[i]) for i in range(n)])
    cx, cy, cz = cart[:, 0], cart[:, 1], cart[:, 2]

    deltas = np.diff(cart, axis=0)
    seg_len = np.linalg.norm(deltas, axis=1)
    seg_is_ext = is_ext[1:]
    ext_len = float(seg_len[seg_is_ext].sum())
    travel_len = float(seg_len[~seg_is_ext].sum())

    z_jumps_travel = int(np.sum((~seg_is_ext) & (np.abs(deltas[:, 2]) > Z_JUMP_MM)))
    z_jumps_extrude = int(np.sum(seg_is_ext & (np.abs(deltas[:, 2]) > Z_JUMP_MM)))
    long_travels = int(np.sum((~seg_is_ext) & (seg_len > LONG_TRAVEL_MM)))

    near_singularity = int(np.sum(np.abs(B) < B_SINGULARITY_DEG))

    b_hist, b_edges = np.histogram(B, bins=10)

    layer_height_guess = 0.2
    zmin, zmax = float(cz.min()), float(cz.max())
    n_layers = max(1, int(np.ceil((zmax - zmin) / layer_height_guess)))
    layer_idx = np.clip(((cz[1:] - zmin) / layer_height_guess).astype(int), 0, n_layers - 1)
    per_layer_ext = np.zeros(n_layers)
    for i, length in zip(layer_idx, seg_len * seg_is_ext):
        per_layer_ext[i] += length
    empty_layers = int(np.sum(per_layer_ext[:-1] == 0))

    def rng(arr):
        return {"min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean())}

    return {
        "empty": False,
        "num_moves": int(n),
        "num_extrude": int(is_ext.sum()),
        "num_travel": int((~is_ext).sum()),
        "extrude_length_mm": round(ext_len, 3),
        "travel_length_mm": round(travel_len, 3),
        "travel_to_extrude_ratio": round(travel_len / ext_len, 4) if ext_len > 0 else None,
        "machine_bounds": {
            "X": rng(X), "Z": rng(Z), "C": rng(C), "B": rng(B),
        },
        "cartesian_bounds": {
            "x": rng(cx), "y": rng(cy), "z": rng(cz),
        },
        "b_axis": {
            "range_deg": [float(B.min()), float(B.max())],
            "histogram": [int(v) for v in b_hist],
            "bin_edges": [round(float(e), 2) for e in b_edges],
            "near_singularity_count": near_singularity,
        },
        "c_axis": {
            "range_deg": [float(C.min()), float(C.max())],
            "span_deg": float(C.max() - C.min()),
        },
        "z_jumps_in_travel": z_jumps_travel,
        "z_jumps_in_extrude": z_jumps_extrude,
        "long_travel_moves": long_travels,
        "layers": {
            "assumed_height_mm": layer_height_guess,
            "count": n_layers,
            "empty_layer_count": empty_layers,
            "ext_per_layer_mm": [round(float(v), 2) for v in per_layer_ext],
        },
    }


def diff_stats(new, base):
    if new.get("empty") or base.get("empty"):
        return {"note": "one side empty, skipping diff"}

    def delta(path, a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return round(a - b, 4)
        return None

    out = {}
    for key in ("num_moves", "num_extrude", "num_travel", "extrude_length_mm",
                "travel_length_mm", "z_jumps_in_travel", "z_jumps_in_extrude",
                "long_travel_moves"):
        out[key] = delta(key, new.get(key), base.get(key))

    out["b_range_delta"] = [
        round(new["b_axis"]["range_deg"][0] - base["b_axis"]["range_deg"][0], 3),
        round(new["b_axis"]["range_deg"][1] - base["b_axis"]["range_deg"][1], 3),
    ]
    out["cartesian_z_max_delta"] = round(
        new["cartesian_bounds"]["z"]["max"] - base["cartesian_bounds"]["z"]["max"], 3
    )
    out["empty_layers_delta"] = (
        new["layers"]["empty_layer_count"] - base["layers"]["empty_layer_count"]
    )
    return out


def print_summary(stats, diff=None):
    if stats.get("empty"):
        print("EMPTY: no G0/G1 moves parsed")
        return

    cb = stats["cartesian_bounds"]
    mb = stats["machine_bounds"]
    print(f"moves: {stats['num_moves']} "
          f"(extrude {stats['num_extrude']}, travel {stats['num_travel']})")
    print(f"extrude: {stats['extrude_length_mm']:.1f} mm | "
          f"travel: {stats['travel_length_mm']:.1f} mm | "
          f"ratio: {stats['travel_to_extrude_ratio']}")
    print(f"cartesian: x[{cb['x']['min']:.1f},{cb['x']['max']:.1f}] "
          f"y[{cb['y']['min']:.1f},{cb['y']['max']:.1f}] "
          f"z[{cb['z']['min']:.1f},{cb['z']['max']:.1f}]")
    print(f"machine:   X[{mb['X']['min']:.1f},{mb['X']['max']:.1f}] "
          f"Z[{mb['Z']['min']:.1f},{mb['Z']['max']:.1f}] "
          f"C[{mb['C']['min']:.1f},{mb['C']['max']:.1f}] "
          f"B[{mb['B']['min']:.1f},{mb['B']['max']:.1f}]")
    print(f"B-axis range: {stats['b_axis']['range_deg']} "
          f"near_singularity={stats['b_axis']['near_singularity_count']}")
    print(f"flags: z_jumps_in_travel={stats['z_jumps_in_travel']} "
          f"z_jumps_in_extrude={stats['z_jumps_in_extrude']} "
          f"long_travels={stats['long_travel_moves']} "
          f"empty_layers={stats['layers']['empty_layer_count']}/{stats['layers']['count']}")

    if diff is not None:
        print("\ndiff vs baseline:")
        for k, v in diff.items():
            print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcode", required=True)
    parser.add_argument("--baseline", help="Path to a previous stats.json for diffing")
    parser.add_argument("--out", help="Directory to write stats.json (default: print only)")
    args = parser.parse_args()

    if not os.path.exists(args.gcode):
        print(f"not found: {args.gcode}", file=sys.stderr)
        sys.exit(1)

    moves = parse_gcode(args.gcode)
    stats = compute_stats(moves)
    stats["source"] = os.path.abspath(args.gcode)

    diff = None
    if args.baseline and os.path.exists(args.baseline):
        with open(args.baseline) as f:
            base = json.load(f)
        diff = diff_stats(stats, base)
        stats["diff_vs"] = os.path.abspath(args.baseline)
        stats["diff"] = diff

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"wrote {out_dir / 'stats.json'}")

    print_summary(stats, diff)


if __name__ == "__main__":
    main()
