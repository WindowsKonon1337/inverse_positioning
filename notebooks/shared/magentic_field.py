import math

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

EARTH_RADIUS_M = 6_371_000.0
STEP_METERS = 500.0
STEP_RAD = STEP_METERS / EARTH_RADIUS_M


def normalize(x, eps=1e-8):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def sample_uniform_sphere(n, device=DEVICE):
    return normalize(torch.randn(n, 3, device=device, dtype=DTYPE))


def tangent_basis(u):
    flat = u.reshape(-1, 3)
    z = torch.tensor([0.0, 0.0, 1.0], device=flat.device, dtype=flat.dtype).expand_as(
        flat
    )
    x = torch.tensor([1.0, 0.0, 0.0], device=flat.device, dtype=flat.dtype).expand_as(
        flat
    )
    ref = torch.where(flat[:, 2:3].abs() < 0.9, z, x)
    e1 = normalize(torch.cross(ref, flat, dim=-1))
    e2 = torch.cross(flat, e1, dim=-1)
    return e1.reshape_as(u), e2.reshape_as(u)


def exp_map_sphere(u, tangent_step_meters):
    step_meters = tangent_step_meters.norm(dim=-1, keepdim=True)
    theta = step_meters / EARTH_RADIUS_M
    direction = tangent_step_meters / step_meters.clamp_min(1e-8)
    moved = torch.cos(theta) * u + torch.sin(theta) * direction
    return normalize(moved)


FIELD_CENTERS = normalize(
    torch.tensor(
        [
            [0.22, -0.61, 0.76],
            [-0.78, 0.34, 0.52],
            [0.57, 0.73, -0.38],
            [-0.18, -0.83, -0.53],
        ],
        dtype=DTYPE,
    )
)
FIELD_MOMENTS = normalize(
    torch.tensor(
        [
            [0.88, 0.17, -0.44],
            [-0.31, 0.92, 0.24],
            [0.12, -0.58, 0.81],
            [-0.70, -0.21, -0.68],
        ],
        dtype=DTYPE,
    )
)
FIELD_STRENGTHS = torch.tensor([1.1, -0.8, 0.65, -0.55], dtype=DTYPE)
WAVE_DIRS = normalize(
    torch.tensor(
        [
            [1.0, 0.3, -0.2],
            [-0.4, 1.0, 0.5],
            [0.2, -0.7, 1.0],
            [0.8, -0.1, 0.6],
            [-0.5, -0.6, 0.7],
            [0.3, 0.9, 0.4],
        ],
        dtype=DTYPE,
    )
)
FIELD_FEATURE_SCALES_M = torch.tensor(
    [120_000.0, 60_000.0, 24_000.0, 12_000.0, 6_000.0, 3_000.0, 1_500.0, 750.0],
    dtype=DTYPE,
)
FIELD_FEATURE_WEIGHTS = torch.tensor(
    [0.08, 0.075, 0.065, 0.055, 0.045, 0.035, 0.027, 0.020], dtype=DTYPE
)


def B(u):
    """Synthetic magnetic field B(u) -> (Bx, By, Bz) for unit 3D vector u."""
    shape = u.shape
    flat = normalize(u.reshape(-1, 3))
    centers = FIELD_CENTERS.to(flat.device)
    moments = FIELD_MOMENTS.to(flat.device)
    strengths = FIELD_STRENGTHS.to(flat.device)
    dirs = WAVE_DIRS.to(flat.device)
    scales = FIELD_FEATURE_SCALES_M.to(flat.device)
    weights = FIELD_FEATURE_WEIGHTS.to(flat.device)

    r = flat[:, None, :] - 0.55 * centers[None, :, :]
    r2 = (r * r).sum(dim=-1).clamp_min(1e-4)
    mr = (moments[None, :, :] * r).sum(dim=-1)
    dipoles = strengths[None, :, None] * (
        3.0 * r * mr[:, :, None] / r2[:, :, None] ** 2.5
        - moments[None, :, :] / r2[:, :, None] ** 1.5
    )
    dipole_field = dipoles.sum(dim=1)

    projected_m = EARTH_RADIUS_M * (flat @ dirs.T)
    fine = torch.zeros_like(flat)
    for j, (scale, weight) in enumerate(zip(scales, weights, strict=True)):
        p0 = 2.0 * math.pi * projected_m[:, j % dirs.shape[0]] / scale
        p1 = 2.0 * math.pi * projected_m[:, (j + 2) % dirs.shape[0]] / scale
        p2 = 2.0 * math.pi * projected_m[:, (j + 4) % dirs.shape[0]] / scale
        fine[:, 0] = fine[:, 0] + weight * torch.sin(p0 + 0.37 * j)
        fine[:, 1] = fine[:, 1] + weight * torch.cos(p1 + 0.53 * j)
        fine[:, 2] = fine[:, 2] + weight * torch.sin(p2 - 0.29 * j)

    # A weak radial backbone keeps the problem globally grounded, while the
    # kilometer-scale texture makes longer trajectories much more informative
    # than a single local magnetic sample.
    field = 0.22 * flat + 0.020 * dipole_field + 0.55 * fine
    return field.reshape(shape)


def fibonacci_sphere(n, device=DEVICE):
    i = torch.arange(n, device=device, dtype=DTYPE) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = math.pi * (3.0 - math.sqrt(5.0)) * i
    r = torch.sqrt((1.0 - z * z).clamp_min(0.0))
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=-1)
