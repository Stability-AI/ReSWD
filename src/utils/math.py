import math
from typing import Optional, Union

import torch


def _random_orthonormal_matrix(d: int, device: torch.device) -> torch.Tensor:
    """Draw a random rotation matrix Q ∈ SO(d) (Haar) via QR-factorisation."""
    a = torch.randn(d, d, device=device)
    # QR gives orthonormal columns; ensure right-handed
    q, r = torch.linalg.qr(a, mode="reduced")
    # make determinant +1 (special orthogonal) – flip first column if needed
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q  # (d,d)


def sobol_sphere(
    n: int,
    d: int,
    device: torch.device,
    sobol_engine: Optional[torch.quasirandom.SobolEngine] = None,
) -> Union[torch.Tensor, torch.quasirandom.SobolEngine]:
    """n unit vectors on S^{d-1} via scrambled Sobol + Gaussian + random rotation."""
    if sobol_engine is None:
        sob = torch.quasirandom.SobolEngine(dimension=d, scramble=True)
    else:
        sob = sobol_engine
    # Draw in [0,1)^d then map → 𝒩(0,1)
    u01 = sob.draw(n).to(device)

    eps = 1e-7
    u01 = u01.clamp(min=eps, max=1.0 - eps)  # avoid 0 and 1 exactly

    z = torch.erfinv(2.0 * u01 - 1.0) * math.sqrt(2.0)  # inverse-CDF of Normal
    z = z / (z.norm(dim=1, keepdim=True) + 1e-8)  # project to sphere
    # Random global rotation (RQMC) to make estimator unbiased
    Q = _random_orthonormal_matrix(d, device)
    return z @ Q.T, sob  # (n,d)
