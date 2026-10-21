"""Aggregation of COSMO surface segments into fixed-area patches.

Segments are clustered with k-means on their coordinates. The number of
clusters is the cavity area divided by TARGET_AREA, capped at PAD. Each patch
stores the area-weighted centroid, the summed area, area-weighted mean and
variance of the raw and the Mullins-averaged sigma, the area-weighted mean
solute potential, a burial measure, and the area fractions of eight elements.

Feature layout of the resulting (P, 18) array:

    0-2   patch centre xyz, not an encoder input, used for distances only
    3     area
    4     sigma_raw mean
    5     sigma_raw variance
    6     sigma_avg mean
    7     sigma_avg variance
    8     solute potential
    9     solid angle, the auxiliary reconstruction target
    10-17 area fraction of Z in Z_LIST

The twelve encoder inputs are columns 3, 4, 5, 8 and 10 to 17.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

R_AV2 = 0.8176300195 ** 2
TARGET_AREA = 2.1
PAD = 192
N_RAY = 64
Z_LIST = [1, 6, 7, 8, 9, 15, 16, 17]
N_DIM = 18
ENCODER_INPUT_COLS = [3, 4, 5, 8] + list(range(10, 18))
KMEANS_SEED = 42


def sigma_avg(xyz, sigma, area, block=512):
    """Mullins averaging of sigma over the surface with averaging radius R_AV2."""
    rn2 = area / np.pi
    den = rn2[None, :] + R_AV2
    pre = rn2[None, :] * R_AV2 / den
    out = np.empty(len(sigma))
    for start in range(0, len(sigma), block):
        end = min(start + block, len(sigma))
        weights = np.exp(-cdist(xyz[start:end], xyz, "sqeuclidean") / den) * pre
        out[start:end] = (weights * sigma[None, :]).sum(1) / weights.sum(1)
    return out


def fibonacci_hemisphere(n, normal):
    """n Fibonacci directions on the hemisphere around `normal`."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    v = np.c_[np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    axis = np.array([0.0, 0.0, 1.0])
    unit = normal / (np.linalg.norm(normal) + 1e-12)
    cross = np.cross(axis, unit)
    cos = float(np.dot(axis, unit))
    if np.linalg.norm(cross) < 1e-9:
        return v if cos > 0 else -v
    k = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    return v @ (np.eye(3) + k + k @ k * (1 / (1 + cos))).T


def solid_angle(centres, areas, n_ray=N_RAY):
    """Fraction of outward hemisphere directions blocked by other patches."""
    n = len(centres)
    total = areas.sum()
    centroid = (areas[:, None] * centres).sum(0) / total
    radii = np.sqrt(areas / np.pi)
    tree = cKDTree(centres)
    k = min(9, n)
    _, idx = tree.query(centres, k=k)
    blocked = np.zeros(n)
    n_fit = 0
    for i in range(n):
        neighbours = centres[idx[i]] if k > 1 else centres[i][None]
        if len(neighbours) >= 4:
            _, _, vt = np.linalg.svd(neighbours - neighbours.mean(0))
            normal = vt[-1]
            n_fit += 1
        else:
            normal = centres[i] - centroid
        outward = centres[i] - centroid
        if np.dot(normal, outward) < 0:
            normal = -normal
        directions = fibonacci_hemisphere(n_ray, normal / (np.linalg.norm(normal) + 1e-12))
        relative = np.delete(centres, i, 0) - centres[i]
        other_radii = np.delete(radii, i)
        if len(relative) == 0:
            continue
        along = relative @ directions.T
        perpendicular = (relative ** 2).sum(1)[:, None] - along ** 2
        blocked[i] = ((along > 0) & (perpendicular < (other_radii ** 2)[:, None])).any(0).mean()
    return blocked, n_fit / n


def build_patches(segments, target_area=TARGET_AREA, pad=PAD, n_ray=N_RAY):
    """Aggregate an (n_segment, 7) segment array into an (P, 18) patch array.

    `segments` columns follow cosmo_io.SEGMENT_COLUMNS. Returns the patch array,
    the number of patches, whether the cluster count was capped at `pad`, and
    the fraction of patches whose normal came from an SVD fit.
    """
    xyz = segments[:, :3].astype(np.float64)
    sigma = segments[:, 3].astype(np.float64)
    area = segments[:, 4].astype(np.float64)
    potential = segments[:, 5].astype(np.float64)
    z_number = segments[:, 6].astype(int)

    n_segment = len(sigma)
    total_area = area.sum()
    n_cluster = max(2, min(int(round(total_area / target_area)), n_segment, pad))
    capped = int(round(total_area / target_area)) > pad

    labels = KMeans(n_clusters=n_cluster, n_init=1, random_state=KMEANS_SEED,
                    max_iter=50).fit(xyz).labels_
    averaged = sigma_avg(xyz, sigma, area)

    rows = []
    for cluster in range(n_cluster):
        take = labels == cluster
        if not take.any():
            continue
        a = area[take]
        a_sum = a.sum()
        if a_sum <= 0:
            continue
        centre = (a[:, None] * xyz[take]).sum(0) / a_sum
        m_raw = (a * sigma[take]).sum() / a_sum
        v_raw = (a * (sigma[take] - m_raw) ** 2).sum() / a_sum
        m_avg = (a * averaged[take]).sum() / a_sum
        v_avg = (a * (averaged[take] - m_avg) ** 2).sum() / a_sum
        pot = (a * potential[take]).sum() / a_sum
        fractions = [a[z_number[take] == z].sum() / a_sum for z in Z_LIST]
        rows.append([centre[0], centre[1], centre[2], a_sum, m_raw, v_raw,
                     m_avg, v_avg, pot, 0.0] + fractions)

    patches = np.asarray(rows, dtype=np.float64)
    n_patch = len(patches)
    blocked, fit_fraction = solid_angle(patches[:, :3], patches[:, 3], n_ray=n_ray)
    patches[:, 9] = blocked
    return patches.astype(np.float32), n_patch, capped, float(fit_fraction)


def pad_patches(patches, pad=PAD, n_dim=N_DIM):
    """Pad a patch array to (pad, n_dim) and return it with a validity mask."""
    out = np.zeros((pad, n_dim), np.float32)
    mask = np.zeros(pad, bool)
    n = min(len(patches), pad)
    out[:n] = patches[:n]
    mask[:n] = True
    return out, mask
