"""Reader for DMol3-style .cosmo files.

The segment table is introduced by a line containing "position (X, Y, Z)" and
holds nine columns:

    n  atom  X  Y  Z  charge  area  charge/area  potential

Coordinates are in atomic units and are converted to Angstrom. Charge, area and
potential are taken as written. Element symbols come from the
"Atom number and name" block and are mapped to atomic numbers.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

BOHR_TO_A = 0.529177
ZERO_AREA = 5e-6

ELEMENT_Z = {"H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Na": 11, "Mg": 12,
             "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "K": 19, "Ca": 20,
             "Br": 35, "I": 53}

_ATOM_LINE = re.compile(r"\s*(\d+)([A-Z][a-z]?)\d*\s+([\d.]+)")
_ATOM_BLOCK = re.compile(r"Atom number and name.*?\n(.*?)\n\s*=+", re.DOTALL)

SEGMENT_COLUMNS = ("x_A", "y_A", "z_A", "sigma_e_per_A2", "area_A2", "potential", "Z")


def atom_elements(text: str) -> dict[int, str]:
    """Map atom index to element symbol using the atom block of a .cosmo file."""
    match = _ATOM_BLOCK.search(text)
    if not match:
        return {}
    elements = {}
    for line in match.group(1).strip().splitlines():
        m = _ATOM_LINE.match(line)
        if m:
            elements[int(m.group(1))] = m.group(2)
    return elements


def read_cosmo(path: str | Path, center: bool = True, drop_zero_area: bool = True):
    """Read one .cosmo file into an (n_segment, 7) array.

    Columns follow SEGMENT_COLUMNS. Sigma is charge divided by area. When
    `center` is set the surface is translated so that the area-weighted centroid
    sits at the origin, which is what the pre-training corpus uses.

    Returns None when the file holds no segment table.
    """
    path = Path(path)
    text = path.read_text()
    elements = atom_elements(text)

    rows = []
    in_segments = False
    for line in text.splitlines():
        if "position (X, Y, Z)" in line:
            in_segments = True
            continue
        if not in_segments:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("="):
            if rows:
                break
            continue
        parts = stripped.split()
        if len(parts) < 9:
            continue
        try:
            atom = int(parts[1])
            x, y, z = (float(parts[i]) * BOHR_TO_A for i in (2, 3, 4))
            charge = float(parts[5])
            area = float(parts[6])
            potential = float(parts[8])
        except ValueError:
            continue
        z_number = ELEMENT_Z.get(elements.get(atom, ""), 0)
        rows.append([x, y, z, charge, area, potential, float(z_number)])

    if not rows:
        return None

    raw = np.asarray(rows, dtype=np.float64)
    if drop_zero_area:
        raw = raw[raw[:, 4] >= ZERO_AREA]
    if len(raw) == 0:
        return None

    xyz, charge, area = raw[:, :3], raw[:, 3], raw[:, 4]
    if center:
        xyz = xyz - (xyz * area[:, None]).sum(0) / area.sum()
    sigma = charge / area
    out = np.column_stack([xyz, sigma, area, raw[:, 5], raw[:, 6]])
    return out.astype(np.float32)


def cavity_area(text: str) -> float:
    """Total cavity area in Angstrom^2 as reported in the file header."""
    m = re.search(r"Total surface area of cavity \(A\*\*2\)\s*=\s*([\d.eE+-]+)", text)
    return float(m.group(1)) if m else float("nan")


def cavity_volume(text: str) -> float:
    """Total cavity volume in Angstrom^3 as reported in the file header."""
    m = re.search(r"Total volume of cavity \(A\*\*3\)\s*=\s*([\d.eE+-]+)", text)
    return float(m.group(1)) if m else float("nan")
