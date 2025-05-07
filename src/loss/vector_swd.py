import typing

import torch
import torch.nn.functional as F
from jaxtyping import Float

from src.loss.abstract_loss import AbstractLoss
from src.utils.math import sobol_sphere


def process_vector(
    x: Float[torch.Tensor, "B D N"],
    dirs: Float[torch.Tensor, "K D"],
) -> Float[torch.Tensor, "K B*N_valid"]:
    """
    Project a 1-D sequence with a bank of linear directions.

    Args
    ----
    x      : (B, D, N) tensor – predictions or ground truth
    dirs   : (K, D)   tensor – unit-length projection directions

    Returns
    -------
    proj   : (K, B*N_valid) tensor of flattened projections
    """
    B, D, N = x.shape
    K, _ = dirs.shape

    # linear projection:   x  (B,D,N)  ->  (B,N,K)  ->  (K,B*N)
    proj = F.linear(x.transpose(1, 2).to(torch.float32), dirs.to(torch.float32))
    proj = proj.permute(2, 0, 1).reshape(K, -1).to(x.dtype)

    return proj


class VectorSWDLoss(AbstractLoss):
    """
    1-D Sliced-Wasserstein Distance on sequences.

    This loss computes the sliced Wasserstein distance between predicted and ground
    truth sequences by projecting them onto random directions and computing the
    Wasserstein distance in 1D. It supports reservoir sampling for adaptive direction
    selection and various variance reduction techniques.

    Parameters
    ----------
    num_proj : int, default=64
        Number of random projections to use per step (K).

    distance : {"l1", "l2"}, default="l1"
        Distance metric to use for computing the Wasserstein distance.

    use_ucv : bool, default=False
        Whether to use upper bounds control variates for variance reduction.
        Mutually exclusive with use_lcv.

    use_lcv : bool, default=False
        Whether to use lower bounds control variates for variance reduction.
        Mutually exclusive with use_ucv.

    refresh_projections_every_n_steps : int, default=1
        How often to refresh the projection directions. A value of 1 means
        refresh every step, higher values reuse directions for multiple steps.

    num_new_candidates : int, default=16
        Number of new candidate directions to generate per step (M).
        If 0, reservoir sampling is disabled. Must not exceed num_proj.

    ess_alpha : float, default=0.5
        Effective sample size threshold for resetting the reservoir.
        When ESS drops below ess_alpha * reservoir_size, the reservoir is reset.

    time_decay_tau : float or None, default=30.0
        Time decay parameter for reservoir weights. If None, no time decay is applied.
        Weights decay exponentially with age: exp(-age / time_decay_tau).

    missing_value_method : {"random_replicate", "interpolate"},
            default="random_replicate"
        Method for handling sequences of different lengths:
        - "random_replicate": Randomly replicate shorter sequences
        - "interpolate": Use linear interpolation to match lengths

    sampling_mode : {"gaussian", "qmc"}, default="qmc"
        Method for generating random projection directions:
        - "gaussian": Standard Gaussian sampling
        - "qmc": Quasi-Monte Carlo sampling using Sobol sequences

    Notes
    -----
    - Reservoir sampling is enabled when num_new_candidates > 0
    - Reservoir size = num_proj - num_new_candidates
    - When use_ucv or use_lcv is True, variance reduction is applied using
      control variates based on the difference between sample and population means
    - The loss automatically handles sequences of different lengths using the
      specified missing_value_method
    """

    def __init__(
        self,
        num_proj: int = 64,
        distance: typing.Literal["l1", "l2"] = "l1",
        use_ucv: bool = False,
        use_lcv: bool = False,
        refresh_projections_every_n_steps: int = 1,
        num_new_candidates: int = 16,
        ess_alpha: float = 0.5,
        time_decay_tau: float | None = 30.0,
        missing_value_method: typing.Literal[
            "random_replicate", "interpolate"
        ] = "random_replicate",
        sampling_mode: typing.Literal[
            "gaussian",
            "qmc",
        ] = "qmc",
    ):
        super().__init__()

        assert not (use_ucv and use_lcv), "use_ucv and use_lcv cannot both be True"

        self.num_proj = num_proj
        self.distance = distance
        self.use_ucv = use_ucv
        self.use_lcv = use_lcv

        self.refresh_projections_every_n_steps = refresh_projections_every_n_steps
        self.num_new_candidates = num_new_candidates  # M
        self.ess_alpha = ess_alpha
        self.time_decay_tau = time_decay_tau
        self.missing_value_method = missing_value_method

        if num_new_candidates > 0 and self.refresh_projections_every_n_steps != 1:
            # Print a warning that this is not recommended
            print(
                "WARNING: num_new_candidates > 0 (enabling reservoir sampling) and "
                "refresh_projections_every_n_steps != 1 is not recommended"
            )
        assert (
            num_new_candidates <= num_proj
        ), "`num_new_candidates` must not exceed `num_proj`"

        # internal state for reservoir sampling
        self.restir_enabled = self.num_new_candidates > 0
        self.reservoir_size = self.num_proj - self.num_new_candidates
        self.register_buffer("_reservoir_filters", torch.empty(0))
        self.register_buffer("_reservoir_weights", torch.empty(0))
        self.register_buffer("_reservoir_steps", torch.empty(0, dtype=torch.long))
        self.register_buffer("_reservoir_keys", torch.empty(0))
        self.register_buffer("_cumulative_weights", torch.tensor(0.0))
        self.register_buffer("_has_reservoir", torch.tensor(False, dtype=torch.bool))

        self._cached_dirs: typing.Optional[torch.Tensor] = None
        self.sampling_mode = sampling_mode
        self.sobol_engine = None

    def _gaussian_proposals(self, k: int, d: int, device: torch.device) -> torch.Tensor:
        """Generate Gaussian random projection directions."""
        w = torch.randn(k, d, device=device)
        return w / (w.norm(dim=1, keepdim=True) + 1e-8)  # unit length

    def _qmc_proposals(self, k: int, d: int, device: torch.device) -> torch.Tensor:
        """Generate quasi-Monte Carlo projection directions using Sobol sequences."""
        vecs, self.sobol_engine = sobol_sphere(k, d, device, self.sobol_engine)
        return vecs.view(k, d)

    def _draw_dirs(self, k: int, d: int, device: torch.device) -> torch.Tensor:
        """Draw projection directions using the specified sampling mode."""
        if self.sampling_mode == "gaussian":
            return self._gaussian_proposals(k, d, device)
        if self.sampling_mode == "qmc":
            return self._qmc_proposals(k, d, device)
        raise ValueError("bad sampling_mode")

    @staticmethod
    def _duplicate_to_match(a: torch.Tensor, b: torch.Tensor, method: str):
        """
        Make two tensors have the same length by duplicating the shorter one.

        Args
        ----
        a, b : (K, N₁) and (K, N₂) tensors
        method : "random_replicate" or "interpolate"

        Returns
        -------
        a, b : Tensors with matching second dimension
        """
        if a.shape[1] == b.shape[1]:
            return a, b
        if a.shape[1] < b.shape[1]:
            a, b = b, a  # swap so that `a` is the larger

        K, NA = a.shape
        NB = b.shape[1]

        # repeat / interpolate B until it matches A
        if method == "random_replicate":
            repeats = NA // NB
            b = torch.cat([b] * repeats, dim=1)
            if b.shape[1] < NA:
                idx = torch.randint(0, NB, (NA - b.shape[1],), device=b.device)
                b = torch.cat([b, b[:, idx]], dim=1)
        else:  # interpolate
            b = F.interpolate(
                b.unsqueeze(0), size=(NA,), mode="linear", align_corners=False
            ).squeeze(0)
        return a, b

    def reset(self):
        """Reset the reservoir sampling state."""
        if self.restir_enabled:
            self._reservoir_filters = torch.empty(0)
            self._reservoir_weights = torch.empty(0)
            self._cumulative_weights.data.fill_(0)
            self._has_reservoir.fill_(False)
            self._reservoir_steps = torch.empty(0, dtype=torch.long)
            self._reservoir_keys = torch.empty(0)

    def _wrs_multi(
        self, filters: torch.Tensor, weights: torch.Tensor, step: int
    ) -> torch.Tensor:
        """
        Weighted reservoir sampling that keeps exactly self.reservoir_size samples and
        returns their indices inside the concatenated candidate set.

        Args
        ----
        filters : (K+M, D) tensor of candidate directions
        weights : (K+M,) tensor of importance weights
        step : Current training step

        Returns
        -------
        keep_idx : Indices of kept samples
        keep_w : Normalized weights of kept samples
        """
        R = self.reservoir_size
        device = weights.device

        u = torch.rand_like(weights)
        keys = u.pow(1.0 / weights.clamp_min(1e-9))

        if not self._has_reservoir.item():
            self._reservoir_filters = filters[:R]
            self._reservoir_weights = weights[:R]
            self._reservoir_keys = keys[:R]
            self._reservoir_steps = torch.full(
                (R,), step, dtype=torch.long, device=device
            )
            self._has_reservoir.fill_(True)

        new_filters = filters[R:]
        new_keys = keys[R:]
        new_weights = weights[R:]
        new_steps = torch.full(
            (new_filters.size(0),), step, dtype=torch.long, device=device
        )

        all_filters = torch.cat([self._reservoir_filters, new_filters], 0)
        all_keys = torch.cat([self._reservoir_keys, new_keys], 0)
        all_weights = torch.cat([self._reservoir_weights, new_weights], 0)
        all_steps = torch.cat([self._reservoir_steps, new_steps], 0)

        topk_keys, topk_idx = torch.topk(all_keys, R, largest=True)

        self._reservoir_filters = all_filters[topk_idx]
        self._reservoir_weights = all_weights[topk_idx]
        self._reservoir_keys = topk_keys
        self._reservoir_steps = all_steps[topk_idx]

        # indices w.r.t. current cand_dirs (old R first, then new M)
        keep_idx = torch.cat(
            [
                torch.arange(R, device=device),
                torch.arange(R, R + new_filters.size(0), device=device),
            ]
        )[topk_idx]
        keep_w = self._reservoir_weights / self._reservoir_weights.sum().clamp_min(
            1e-12
        )
        return keep_idx, keep_w

    def _apply_time_decay(self, step: int):
        """
        Apply exponential time decay to stored reservoir weights.

        Args
        ----
        step : Current training step
        """
        if self.time_decay_tau is None or not self._has_reservoir.item():
            return
        age = (step - self._reservoir_steps).to(torch.float32)
        decay = torch.exp(-age / self.time_decay_tau).to(self._reservoir_weights.dtype)
        self._reservoir_weights.mul_(decay)
        self._reservoir_keys.mul_(decay)  # preserve ordering consistency

    def forward(
        self,
        pred: Float[torch.Tensor, "B D N"],
        gt: Float[torch.Tensor, "B D N"],
        step: int,
    ):
        """
        Compute the sliced Wasserstein distance between predicted and ground truth
        sequences.

        Args
        ----
        pred : (B, D, N) tensor of predicted sequences
        gt : (B, D, N) tensor of ground truth sequences
        step : Current training step for reservoir sampling

        Returns
        -------
        loss : Scalar tensor containing the computed loss
        """
        B, D, N = pred.shape
        K = self.num_proj
        M = self.num_new_candidates
        R = self.reservoir_size
        device = pred.device
        gt = gt.detach()

        self._apply_time_decay(step)

        # Get candidate directions
        if step % self.refresh_projections_every_n_steps == 0:
            new_dirs = self._draw_dirs(
                M if self.restir_enabled and self._has_reservoir.item() else K,
                D,
                device,
            )
            self._cached_dirs = new_dirs
        else:
            new_dirs = self._cached_dirs

        if self.restir_enabled and self._has_reservoir.item():
            cand_dirs = torch.cat(
                [self._reservoir_filters, new_dirs], dim=0
            )  # [K+M, C,P,P]
        else:
            cand_dirs = new_dirs

        # Project sequences
        cand_pred = process_vector(pred, cand_dirs)
        cand_gt = process_vector(gt, cand_dirs)

        cand_pred, cand_gt = self._duplicate_to_match(
            cand_pred, cand_gt, self.missing_value_method
        )

        cand_pred = cand_pred.sort(dim=1).values
        cand_gt = cand_gt.sort(dim=1).values

        # Select K directions (reservoir) & importance weights
        if self.restir_enabled:
            with torch.no_grad():
                base = cand_pred - cand_gt
                base = base.abs() if self.distance == "l1" else base.square()
                ris_weights = base.mean(1)  # (K+M)
                keep_idx, keep_w = self._wrs_multi(cand_dirs, ris_weights, step)

            w = keep_w
            w_hat = keep_w

            dirs = cand_dirs[keep_idx]
            proj_pred = cand_pred[keep_idx]
            proj_gt = cand_gt[keep_idx]
        else:
            dirs = cand_dirs
            proj_pred = cand_pred
            proj_gt = cand_gt
            w = torch.full((dirs.shape[0],), 1.0 / K, device=device)

        # Compute SWD
        diff = proj_pred - proj_gt
        diff = diff.abs() if self.distance == "l1" else diff.square()
        per_slice = diff.mean(1)  # (L,)

        if self.use_ucv or self.use_lcv:
            X_vecs = pred.permute(0, 2, 1).reshape(-1, D)  # (B·N, D)
            Y_vecs = gt.permute(0, 2, 1).reshape(-1, D)  # (B·N, D)

            m1 = X_vecs.mean(0)  # (D,)
            m2 = Y_vecs.mean(0)
            diff_m = m1 - m2  # (D,)

            theta = dirs  # (L, D) already unit-norm

            if self.use_ucv:
                diff_X = X_vecs - m1
                diff_Y = Y_vecs - m2

                d = D
                trSigX = diff_X.pow(2).mean()
                trSigY = diff_Y.pow(2).mean()
                G_bar = (diff_m @ diff_m) / d + (trSigX + trSigY)

                delta2 = (theta @ diff_m) ** 2  # (L,)

                proj_X = diff_X @ theta.t()  # (B·N, L)
                proj_Y = diff_Y @ theta.t()
                varX = proj_X.pow(2).mean(0)  # (L,)
                varY = proj_Y.pow(2).mean(0)
                G_hat = delta2 + varX + varY
            else:  # LCV
                d = D
                G_bar = (diff_m @ diff_m) / d
                G_hat = (theta @ diff_m) ** 2

            diff_hat_G_mean_G = G_hat - G_bar

            hat_A = (w * per_slice).sum()
            var_G = (w * diff_hat_G_mean_G.pow(2)).sum()
            cov_AG = (w * (per_slice - hat_A) * diff_hat_G_mean_G).sum()
            hat_alpha = cov_AG / (var_G + 1e-12)
            loss = hat_A - hat_alpha * (w * diff_hat_G_mean_G).sum()
        else:
            loss = (w * per_slice).sum()

        # Reservoir update
        if self.restir_enabled and self.ess_alpha > 0:
            with torch.no_grad():
                ess = (w_hat.sum().square()) / (w_hat.square().sum() + 1e-12)
                ess = torch.nan_to_num(ess, nan=0.0, posinf=R, neginf=0.0).item()
                if ess < self.ess_alpha * R:
                    print(f"ESS: {ess} is less than {self.ess_alpha * R}, resetting")
                    self.reset()

        return loss
