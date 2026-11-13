"""
learned_scheduler.py
═══════════════════════════════════════════════════════════════════════════════
Learning-based adaptive timestep & surrogate-gating scheduler for HELICS EMT/TS
co-simulation.

Replaces fixed 10ms timestamps and ACES-style 2nd/3rd-derivative inflection
checks with:
  1. BoundarySurrogateRLS: An online Recursive Least Squares (RLS) surrogate
     with analytical epistemic uncertainty (phi^T P phi) that learns component
     (P, Q) responses across varying boundary voltages and scenario embeddings.
  2. AdaptiveStepPolicy: A learnable, derivative-free step-size and EMT-bypass
     policy driven by prediction surprise, epistemic uncertainty, normalized
     system state innovation, and scenario context.
═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field, asdict
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SchedulerConfig:
    """Configuration for learned adaptive co-simulation scheduling."""
    mode: str = "learned_adaptive"      # "learned_adaptive" or "fixed"
    dt_fixed_s: float = 10e-3
    dt_min_s: float = 2e-3
    dt_max_s: float = 80e-3
    dt_init_s: float = 10e-3
    forgetting_factor: float = 0.985
    prior_reg: float = 1.0
    surprise_tol_mw: float = 3.5        # prediction error threshold (MW)
    uncertainty_tol: float = 0.18       # dimensionless RLS epistemic std threshold
    allow_surrogate_bypass: bool = True
    max_surrogate_streak: int = 4       # max consecutive EMT-bypass steps
    event_lookahead_s: float = 2e-3     # clamp step to land cleanly on events
    post_event_tight_s: float = 40e-3   # tight coupling window after an event


@dataclass
class ScenarioContext:
    """Scenario & component descriptors passed from JSON / Claude NL engine."""
    rated_p_mw: float = 150.0
    rated_q_mvar: float = 30.6
    v_init_pu: float = 1.00559
    fault_severity: float = 1.0         # 0.0 (no fault) .. 1.0+ (bolted fault)
    component_id: float = 0.0           # numeric encoding of load/IBR profile
    event_times: List[float] = field(default_factory=list)


class BoundarySurrogateRLS:
    """Online RLS surrogate predicting (P_MW, Q_MVAR) and epistemic uncertainty.

    Uses a shared feature basis phi(x) with independent RLS weight vectors for P
    and Q and a shared inverse covariance matrix P_cov. Because P_cov tracks
    data density in feature space, sqrt(phi^T P_cov phi) rises automatically
    when entering unseen operating regimes or component behaviors.
    """

    def __init__(
        self,
        n_features: int = 10,
        forgetting_factor: float = 0.985,
        prior_reg: float = 1.0,
        p_init: float = 150.0,
        q_init: float = 30.6,
    ):
        self.n_features = n_features
        self.lam = forgetting_factor
        self.prior_reg = prior_reg
        self.p_base = p_init
        self.q_base = q_init

        # Initialize weights to predict zero deviation from base operating point
        self.w_p = np.zeros(n_features, dtype=np.float64)
        self.w_q = np.zeros(n_features, dtype=np.float64)
        self.P_cov = np.eye(n_features, dtype=np.float64) / prior_reg
        self.residual_var_p = 1.0
        self.residual_var_q = 1.0
        self.n_updates = 0

    @staticmethod
    def build_features(
        v_mag: float,
        dv: float,
        v_ang_rad: float,
        dangle_rad: float,
        p_prev: float,
        q_prev: float,
        dt: float,
        ctx: ScenarioContext,
    ) -> np.ndarray:
        """Construct bounded, dimensionless feature vector (no 2nd/3rd derivatives)."""
        v_err = v_mag - ctx.v_init_pu
        p_norm = (p_prev - ctx.rated_p_mw) / max(ctx.rated_p_mw, 1.0)
        q_norm = (q_prev - ctx.rated_q_mvar) / max(abs(ctx.rated_q_mvar), 10.0)
        v_sq_drop = max(0.0, 1.0 - v_mag * v_mag)
        return np.array(
            [
                1.0,
                v_err,
                v_err * abs(v_err),
                v_sq_drop,
                np.clip(dv * 10.0, -2.0, 2.0),
                np.clip(dangle_rad * 5.0, -2.0, 2.0),
                np.clip(p_norm, -2.0, 2.0),
                np.clip(q_norm, -2.0, 2.0),
                np.clip(dt * 10.0, 0.0, 2.0),
                np.clip(ctx.component_id * 0.5 + ctx.fault_severity * 0.25, -2.0, 2.0),
            ],
            dtype=np.float64,
        )

    def predict(self, phi: np.ndarray) -> Tuple[float, float, float]:
        """Return (P_hat_MW, Q_hat_MVAR, epistemic_uncertainty)."""
        dp_hat = float(np.clip(float(phi @ self.w_p), -300.0, 300.0))
        dq_hat = float(np.clip(float(phi @ self.w_q), -300.0, 300.0))
        p_hat = max(0.0, min(self.p_base + dp_hat, 500.0))
        q_hat = max(-200.0, min(self.q_base + dq_hat, 300.0))
        epistemic_var = float(np.clip(float(phi @ self.P_cov @ phi), 0.0, 100.0))
        uncertainty = math.sqrt(epistemic_var)
        return p_hat, q_hat, uncertainty

    def update(self, phi: np.ndarray, p_actual: float, q_actual: float) -> Tuple[float, float]:
        """Update RLS weights and covariance from an actual EMT measurement.

        Returns (abs_err_p_mw, abs_err_q_mvar) prior to the update.
        """
        p_hat, q_hat, _ = self.predict(phi)
        err_p = float(np.clip((p_actual - self.p_base) - float(phi @ self.w_p), -200.0, 200.0))
        err_q = float(np.clip((q_actual - self.q_base) - float(phi @ self.w_q), -200.0, 200.0))

        P_phi = self.P_cov @ phi
        denom = self.lam + float(phi @ P_phi)
        gain = P_phi / max(denom, 1e-6)

        # Update weights with physical bounds (+/- 300 MW)
        self.w_p = np.clip(self.w_p + gain * err_p, -300.0, 300.0)
        self.w_q = np.clip(self.w_q + gain * err_q, -300.0, 300.0)

        # Joseph stabilized covariance update (guarantees positive definiteness)
        I_g = np.eye(self.n_features) - np.outer(gain, phi)
        self.P_cov = (I_g @ self.P_cov @ I_g.T + (1e-4) * np.eye(self.n_features)) / self.lam
        self.P_cov = 0.5 * (self.P_cov + self.P_cov.T)
        np.clip(self.P_cov, 1e-4, 50.0, out=self.P_cov)

        alpha = 0.1
        self.residual_var_p = (1.0 - alpha) * self.residual_var_p + alpha * min(err_p ** 2, 1e4)
        self.residual_var_q = (1.0 - alpha) * self.residual_var_q + alpha * min(err_q ** 2, 1e4)
        self.n_updates += 1
        return min(abs(p_actual - p_hat), 100.0), min(abs(q_actual - q_hat), 100.0)

    def to_dict(self) -> Dict:
        return {
            "n_features": self.n_features,
            "lam": self.lam,
            "prior_reg": self.prior_reg,
            "p_base": self.p_base,
            "q_base": self.q_base,
            "w_p": self.w_p.tolist(),
            "w_q": self.w_q.tolist(),
            "P_cov": self.P_cov.tolist(),
            "n_updates": self.n_updates,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BoundarySurrogateRLS":
        obj = cls(
            n_features=data["n_features"],
            forgetting_factor=data["lam"],
            prior_reg=data["prior_reg"],
            p_init=data["p_base"],
            q_init=data["q_base"],
        )
        obj.w_p = np.array(data["w_p"], dtype=np.float64)
        obj.w_q = np.array(data["w_q"], dtype=np.float64)
        obj.P_cov = np.array(data["P_cov"], dtype=np.float64)
        obj.n_updates = int(data.get("n_updates", 0))
        return obj


class AdaptiveStepPolicy:
    """Learned, derivative-free policy selecting (next_dt, skip_emt).

    Replaces fixed 10ms intervals and ACES 2nd/3rd-derivative inflection tests.
    The policy scores system/surrogate state features with learnable weights
    that can be trained offline via `fit_from_trace()` across multiple scenarios.
    """

    def __init__(self, cfg: SchedulerConfig, ctx: ScenarioContext):
        self.cfg = cfg
        self.ctx = ctx
        # Default policy weights for log-step scaling:
        # features = [1.0, norm_surprise, uncertainty, norm_dv, norm_df, post_event_flag]
        # Negative weights shrink dt when surprise/uncertainty/dv/df/event is high.
        self.policy_weights = np.array([1.2, -1.6, -1.3, -1.8, -1.0, -1.5], dtype=np.float64)
        self.surrogate_streak = 0
        self.last_event_t: Optional[float] = None

    def _event_proximity(self, t_now: float) -> Tuple[Optional[float], bool]:
        """Return (distance_to_next_event, in_post_event_window)."""
        next_dist = None
        for et in sorted(self.ctx.event_times):
            if et > t_now + 1e-9:
                next_dist = et - t_now
                break
            if et <= t_now + 1e-9:
                self.last_event_t = et

        in_post_window = (
            self.last_event_t is not None
            and (t_now - self.last_event_t) <= self.cfg.post_event_tight_s
        )
        return next_dist, in_post_window

    @staticmethod
    def build_policy_features(
        surprise_mw: float,
        surprise_tol_mw: float,
        uncertainty: float,
        uncertainty_tol: float,
        dv_pu: float,
        df_hz: float,
        in_post_event: bool,
    ) -> np.ndarray:
        return np.array(
            [
                1.0,
                min(surprise_mw / max(surprise_tol_mw, 0.1), 4.0),
                min(uncertainty / max(uncertainty_tol, 1e-3), 4.0),
                min(abs(dv_pu) / 0.015, 4.0),
                min(abs(df_hz) / 0.05, 4.0),
                1.0 if in_post_event else 0.0,
            ],
            dtype=np.float64,
        )

    def select_action(
        self,
        t_now: float,
        t_end: float,
        surprise_mw: float,
        uncertainty: float,
        dv_pu: float,
        df_hz: float,
    ) -> Tuple[float, bool, Dict[str, float]]:
        """Select (next_dt_s, skip_emt, diagnostics)."""
        remaining = max(0.0, t_end - t_now)
        if remaining <= 1e-9:
            return self.cfg.dt_min_s, False, {"reason": "done"}

        if self.cfg.mode == "fixed":
            dt = min(self.cfg.dt_fixed_s, remaining)
            return dt, False, {"score": 0.0, "surrogate_streak": 0}

        next_event_dist, in_post_event = self._event_proximity(t_now)

        z = self.build_policy_features(
            surprise_mw=surprise_mw,
            surprise_tol_mw=self.cfg.surprise_tol_mw,
            uncertainty=uncertainty,
            uncertainty_tol=self.cfg.uncertainty_tol,
            dv_pu=dv_pu,
            df_hz=df_hz,
            in_post_event=in_post_event,
        )

        # Smooth logistic interpolation in [dt_min, dt_max]
        score = float(z @ self.policy_weights)
        gate = 1.0 / (1.0 + math.exp(-np.clip(score, -10.0, 10.0)))
        log_min = math.log(self.cfg.dt_min_s)
        log_max = math.log(self.cfg.dt_max_s)
        dt = math.exp(log_min + gate * (log_max - log_min))

        # Snap cleanly onto upcoming disturbance events so transitions are exact
        approaching_event = False
        if next_event_dist is not None and next_event_dist <= dt + self.cfg.event_lookahead_s:
            dt = max(min(next_event_dist, dt), 1e-4)
            approaching_event = True

        dt = min(dt, remaining)

        # Decide whether EMT (PSCAD) can be bypassed by the RLS surrogate
        can_bypass = (
            self.cfg.allow_surrogate_bypass
            and not approaching_event
            and not in_post_event
            and surprise_mw < 0.75 * self.cfg.surprise_tol_mw
            and uncertainty < self.cfg.uncertainty_tol
            and abs(dv_pu) < 0.008
            and self.surrogate_streak < self.cfg.max_surrogate_streak
        )

        if can_bypass:
            self.surrogate_streak += 1
        else:
            self.surrogate_streak = 0

        diag = {
            "score": score,
            "gate": gate,
            "surprise_mw": surprise_mw,
            "uncertainty": uncertainty,
            "in_post_event": float(in_post_event),
            "surrogate_streak": float(self.surrogate_streak),
        }
        return dt, can_bypass, diag

    def fit_from_trace(
        self,
        feature_matrix: np.ndarray,
        oracle_dt: np.ndarray,
        l2_reg: float = 1e-2,
    ) -> float:
        """Fit policy weights via ridge regression on oracle safe step sizes."""
        log_min = math.log(self.cfg.dt_min_s)
        log_max = math.log(self.cfg.dt_max_s)
        frac = np.clip((np.log(oracle_dt) - log_min) / max(log_max - log_min, 1e-6), 0.02, 0.98)
        logit_target = np.log(frac / (1.0 - frac))
        A = feature_matrix.T @ feature_matrix + l2_reg * np.eye(feature_matrix.shape[1])
        b = feature_matrix.T @ logit_target
        self.policy_weights = np.linalg.solve(A, b)
        preds = 1.0 / (1.0 + np.exp(-np.clip(feature_matrix @ self.policy_weights, -10.0, 10.0)))
        pred_dt = np.exp(log_min + preds * (log_max - log_min))
        return float(np.sqrt(np.mean((pred_dt - oracle_dt) ** 2)))


def save_learned_state(path: str, surrogate: BoundarySurrogateRLS, policy: AdaptiveStepPolicy) -> None:
    """Persist learned surrogate + step-policy weights across scenarios."""
    payload = {
        "surrogate": surrogate.to_dict(),
        "policy_weights": policy.policy_weights.tolist(),
        "config": asdict(policy.cfg),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_learned_state(
    path: str,
    cfg: SchedulerConfig,
    ctx: ScenarioContext,
) -> Tuple[BoundarySurrogateRLS, AdaptiveStepPolicy]:
    """Load previously trained surrogate + policy weights if available."""
    p = Path(path)
    if not p.exists():
        return (
            BoundarySurrogateRLS(
                forgetting_factor=cfg.forgetting_factor,
                prior_reg=cfg.prior_reg,
                p_init=ctx.rated_p_mw,
                q_init=ctx.rated_q_mvar,
            ),
            AdaptiveStepPolicy(cfg, ctx),
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    surrogate = BoundarySurrogateRLS.from_dict(data["surrogate"])
    # Re-anchor base operating point for new scenario while keeping learned slopes
    surrogate.p_base = ctx.rated_p_mw
    surrogate.q_base = ctx.rated_q_mvar
    policy = AdaptiveStepPolicy(cfg, ctx)
    if "policy_weights" in data:
        policy.policy_weights = np.array(data["policy_weights"], dtype=np.float64)
    return surrogate, policy
