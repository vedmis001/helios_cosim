"""
ts_backend.py
═══════════════════════════════════════════════════════════════════════════════
Transient Stability (TS) backend abstraction for HELICS co-simulation.

Provides a unified single-step interface for:
  1. TSAOptBackend: Wraps tsaopt's `solve_ts_single_step` (trapezoidal DAE
     solver) and `model_dict`/`ix_dict` structures.
  2. PSSEBackend: Wraps PSS/E (`psspy`) from `helios_psse_federate.py`.
  3. SyntheticGridBackend: Built-in pure-NumPy/SciPy trapezoidal DAE test
     grid (synchronous machine + AVR + boundary bus + fault shunt) for rapid
     validation and scheduler benchmarking without PSS/E or full tsaopt data.
═══════════════════════════════════════════════════════════════════════════════
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class TSStepResult:
    """Standardized output after advancing the TS solver by one exchange step."""
    t_s: float
    v_mag_pu: float
    v_angle_deg: float
    freq_hz: float
    min_bus_v_pu: float
    max_rotor_spread_rad: float
    converged: bool
    event_active: bool


class TSBackend(ABC):
    """Abstract single-step TS solver backend for HELICS co-simulation."""

    @abstractmethod
    def initialize(self, scenario: Dict[str, Any]) -> TSStepResult:
        """Initialize the power flow + dynamic state at t=0."""

    @abstractmethod
    def apply_boundary_load(self, p_mw: float, q_mvar: float) -> None:
        """Update the boundary bus load with (P, Q) from EMT or surrogate."""

    @abstractmethod
    def step(self, t_target: float, dt: float) -> TSStepResult:
        """Advance TS dynamics from current time to `t_target` (step `dt`)."""

    @abstractmethod
    def get_event_times(self) -> List[float]:
        """Return known scheduled disturbance timestamps (seconds)."""

    def finalize(self) -> None:
        """Optional cleanup hook."""


class TSAOptBackend(TSBackend):
    """Adapter around `tsaopt.scripts.solvers.transient_stability.solve_ts_single_step`.

    Requires the full `tsaopt` runtime (`settings`, `scripts.stamps_and_residues`,
    `scripts.event_state_manager`, `models.BaseDevice`, `models.simulation`, etc.)
    and a pre-built or case-loaded `(model_dict, ix_dict, V0)`.
    """

    def __init__(
        self,
        tsaopt_root: Optional[str] = None,
        model_dict: Optional[Dict[str, Any]] = None,
        ix_dict: Optional[Dict[str, Any]] = None,
        V0: Optional[np.ndarray] = None,
    ):
        self.tsaopt_root = tsaopt_root or str(Path(__file__).resolve().parent.parent / "tsaopt")
        self.model_dict = model_dict
        self.ix_dict = ix_dict
        self.V_curr = V0.copy() if V0 is not None else None
        self.Ypf = None
        self.boundary_bus_str = "5385"
        self.boundary_bus_idx = 0
        self.boundary_load_mask = None
        self.s_base_mva = 100.0
        self._solve_step_fn = None
        self._set_topology_fn = None
        self._get_event_times_fn = None

    def _ensure_imports(self):
        if self.tsaopt_root not in sys.path:
            sys.path.insert(0, self.tsaopt_root)
        try:
            from scripts.solvers.transient_stability import solve_ts_single_step
            from scripts.stamps_and_residues.transient import generate_Ypf_ts
            from scripts.event_state_manager import (
                initialize_event_state,
                set_topology_at_time,
                get_event_times,
            )
        except ImportError as exc:
            raise RuntimeError(
                "TSAOptBackend requires the remaining tsaopt runtime modules "
                "(settings.py, models/BaseDevice.py, models/simulation/*, "
                "models/variables/*, scripts/stamps_and_residues/*, "
                "scripts/event_state_manager.py, scripts/index_assignment.py, "
                "scripts/limiting/*). Missing import: " + str(exc)
            ) from exc

        self._solve_step_fn = solve_ts_single_step
        self._generate_Ypf_fn = generate_Ypf_ts
        self._init_events_fn = initialize_event_state
        self._set_topology_fn = set_topology_at_time
        self._get_event_times_fn = get_event_times

    def _load_testcase(self, scenario: Dict[str, Any]) -> None:
        """Load and initialize a tsaopt testcase following `entrypoints/run_ts.py`."""
        testcase = scenario.get("testcase")
        if not testcase:
            raise RuntimeError(
                "TSAOptBackend requires either pre-built (model_dict, ix_dict, V0) "
                "or scenario['testcase'] pointing to an .xlsx case file."
            )
        from os import path
        from parsers.parse_tsa import parse_tsa
        from models.prepare_models import prepare_models, check_for_duplicate_pv
        from scripts.index_assignment import (
            assign_indices_ts,
            link_indices,
            check_for_unassigned_indices,
        )
        from scripts.compile_sympy import compile_sympy
        from scripts.initialization.ts_init import solve_init
        from settings import ts_settings

        ts_settings.case_name = path.splitext(path.basename(testcase))[0]
        ts_settings.track_all_rotor_angle_diffs = bool(scenario.get("track_rotor_spread", True))
        data_dict = parse_tsa(testcase)
        model_dict = prepare_models(data_dict, pv_nodyn_model=scenario.get("pv_nodyn_model", "pv"))
        model_dict["bus"].define_soft_voltage_limits()
        ix_dict = assign_indices_ts(model_dict)
        link_indices(model_dict, ix_dict)
        check_for_duplicate_pv(model_dict)
        bad_vars = check_for_unassigned_indices(model_dict)
        if bad_vars:
            raise RuntimeError(f"Unassigned tsaopt indices: {bad_vars}")

        compile_sympy(model_dict, ix_dict, opt=False)
        ts_load_model = "constant_p" if scenario.get("constant_p", False) else "constant_impedance"
        V0 = solve_init(
            model_dict,
            ix_dict,
            ts_load_model=ts_load_model,
            use_tx_stepping=bool(scenario.get("use_tx_stepping", False)),
        )
        self.model_dict = model_dict
        self.ix_dict = ix_dict
        self.V_curr = V0.copy()

    def initialize(self, scenario: Dict[str, Any]) -> TSStepResult:
        self._ensure_imports()
        if self.model_dict is None or self.ix_dict is None or self.V_curr is None:
            self._load_testcase(scenario)

        self.boundary_bus_str = str(scenario.get("boundary_bus", "5385"))
        self.s_base_mva = float(scenario.get("s_base_mva", 100.0))

        bus = self.model_dict["bus"]
        matches = np.where(bus.idx == self.boundary_bus_str)[0]
        self.boundary_bus_idx = int(matches[0]) if len(matches) > 0 else 0

        if "pq" in self.model_dict:
            pq_dev = self.model_dict["pq"]
            mask = pq_dev.bus == self.boundary_bus_str
            if np.any(mask):
                # Ensure the co-simulated boundary load tracks dynamic P/Q injections
                # from PSCAD/surrogate via constant-P terms while leaving other buses untouched.
                pq_dev.k_pow_P[mask] = 1.0
                pq_dev.k_pow_Q[mask] = 1.0
                pq_dev.G_zip[mask] = 0.0
                pq_dev.B_zip[mask] = 0.0
                pq_dev.Ir_zip[mask] = 0.0
                pq_dev.Ii_zip[mask] = 0.0
                self.boundary_load_mask = mask

        if "event_queue" not in self.model_dict:
            self._init_events_fn(self.model_dict, t0=0.0)
        else:
            self._set_topology_fn(self.model_dict, 0.0)

        self.Ypf = self._generate_Ypf_fn(
            self.V_curr[: self.ix_dict["n_alg"]],
            self.model_dict,
            self.ix_dict["n_eq"],
        )
        return self._extract_result(t_s=0.0, converged=True, event_active=False)

    def apply_boundary_load(self, p_mw: float, q_mvar: float) -> None:
        if "pq" in self.model_dict and self.boundary_load_mask is not None and np.any(self.boundary_load_mask):
            pq_dev = self.model_dict["pq"]
            n_match = int(np.sum(self.boundary_load_mask))
            pq_dev.p[self.boundary_load_mask] = (p_mw / self.s_base_mva) / n_match
            pq_dev.q[self.boundary_load_mask] = (q_mvar / self.s_base_mva) / n_match

    def step(self, t_target: float, dt: float) -> TSStepResult:
        events_applied, _ = self._set_topology_fn(self.model_dict, t_target)
        self.Ypf = self._generate_Ypf_fn(
            self.V_curr[: self.ix_dict["n_alg"]],
            self.model_dict,
            self.ix_dict["n_eq"],
        )

        # Sub-step internally if Newton struggles at a large adaptive exchange dt
        V_next, ok = self._solve_step_fn(
            self.model_dict, self.ix_dict, self.V_curr, t_target, dt, self.Ypf
        )
        if not ok:
            n_sub = 4
            sub_dt = dt / n_sub
            V_tmp = self.V_curr.copy()
            t_cur = t_target - dt
            for _ in range(n_sub):
                t_cur += sub_dt
                self._set_topology_fn(self.model_dict, t_cur)
                V_tmp, ok = self._solve_step_fn(
                    self.model_dict, self.ix_dict, V_tmp, t_cur, sub_dt, self.Ypf
                )
                if not ok:
                    break
            V_next = V_tmp

        if ok:
            bus = self.model_dict["bus"]
            if "theta" in bus.alg_vars:
                theta = V_next[bus.theta.i]
                V_next[bus.theta.i] = (theta + math.pi) % (2 * math.pi) - math.pi
            for ctrl in self.model_dict.get("controllers", []):
                ctrl.update_timer_blocks(V_next, t_target)
            self.V_curr = V_next

        return self._extract_result(
            t_s=t_target,
            converged=ok,
            event_active=len(events_applied) > 0,
        )

    def get_event_times(self) -> List[float]:
        if self._get_event_times_fn is None or self.model_dict is None:
            return []
        return [float(x) for x in self._get_event_times_fn(self.model_dict, 0.0, 1e6)]

    def _extract_result(self, t_s: float, converged: bool, event_active: bool) -> TSStepResult:
        bus = self.model_dict["bus"]
        vr = float(self.V_curr[bus.Vr.i[self.boundary_bus_idx]])
        vi = float(self.V_curr[bus.Vi.i[self.boundary_bus_idx]])
        v_mag = math.hypot(vr, vi)
        v_ang_deg = math.degrees(math.atan2(vi, vr))
        vmag_all = np.hypot(self.V_curr[bus.Vr.i], self.V_curr[bus.Vi.i])
        spread = 0.0
        if "rotor_angle_diffs" in self.model_dict:
            spread = float(np.max(np.abs(self.V_curr[self.model_dict["rotor_angle_diffs"].delta_spread.i])))
        freq_hz = 60.0
        for key in ("genrou", "gencls", "simplesg"):
            if key in self.model_dict and hasattr(self.model_dict[key], "omega"):
                omega = float(np.mean(self.V_curr[self.model_dict[key].omega.i]))
                freq_hz = 60.0 * omega
                break
        return TSStepResult(
            t_s=t_s,
            v_mag_pu=v_mag,
            v_angle_deg=v_ang_deg,
            freq_hz=freq_hz,
            min_bus_v_pu=float(np.min(vmag_all)),
            max_rotor_spread_rad=spread,
            converged=converged,
            event_active=event_active,
        )


class PSSEBackend(TSBackend):
    """PSS/E backend delegating to functions in `helios_psse_federate.py`."""

    def __init__(self):
        self._psse = None
        self._t_events: List[float] = []

    def initialize(self, scenario: Dict[str, Any]) -> TSStepResult:
        import helios_psse_federate as psse_mod
        self._psse = psse_mod
        psse_mod.initialize_psse()
        psse_mod.setup_output_channels()
        v_mag, v_ang = psse_mod.get_boundary_voltage()
        dist = scenario.get("disturbance", {})
        for k in ("t_start_s", "t_clear_s"):
            if k in dist and dist[k] is not None:
                self._t_events.append(float(dist[k]))
        return TSStepResult(
            t_s=0.0,
            v_mag_pu=v_mag,
            v_angle_deg=v_ang,
            freq_hz=60.0,
            min_bus_v_pu=v_mag,
            max_rotor_spread_rad=0.0,
            converged=True,
            event_active=False,
        )

    def apply_boundary_load(self, p_mw: float, q_mvar: float) -> None:
        self._psse.apply_datacenter_load(p_mw, q_mvar)

    def step(self, t_target: float, dt: float) -> TSStepResult:
        ierr = self._psse.step_psse(t_target)
        v_mag, v_ang = self._psse.get_boundary_voltage()
        return TSStepResult(
            t_s=t_target,
            v_mag_pu=v_mag,
            v_angle_deg=v_ang,
            freq_hz=60.0,
            min_bus_v_pu=v_mag,
            max_rotor_spread_rad=0.0,
            converged=(ierr == 0),
            event_active=any(abs(t_target - te) < 1e-4 for te in self._t_events),
        )

    def get_event_times(self) -> List[float]:
        return list(self._t_events)


class SyntheticGridBackend(TSBackend):
    """Fast trapezoidal-DAE single-machine + boundary-bus test system.

    Models an equivalent Texas grid generator (swing + governor + AVR voltage
    dynamics) connected through reactance X_eq to Bus 5385 (DALLAS 3 1), where
    the datacenter load (P_dc, Q_dc) and an optional fault shunt G_fault act.
    Integrated with implicit trapezoidal Newton-Raphson just like `solve_ts_single_step`.
    """

    def __init__(self):
        self.s_base_mva = 100.0
        self.x_eq = 0.085
        self.h_const = 4.5
        self.d_damp = 2.0
        self.tau_gov = 0.6
        self.r_droop = 0.05
        self.tau_avr = 0.15
        self.k_avr = 25.0
        self.v_ref = 1.018
        self.omega_0 = 2.0 * math.pi * 60.0

        # State vector x = [delta (rad), omega_pu, e_q (pu), p_m (pu)]
        self.x = np.zeros(4, dtype=np.float64)
        self.v_mag = 1.00559
        self.v_ang_rad = math.radians(-72.489)
        self.p_load_pu = 1.50
        self.q_load_pu = 0.306

        # Disturbance settings
        self.dist_type = "three_phase_fault"
        self.t_start = 1.0
        self.t_clear = 1.0833
        self.g_fault_on = 3.2
        self.load_step_pu = 0.0
        self.v_init_pu = 1.00559
        self.v_init_deg = -72.489

    def initialize(self, scenario: Dict[str, Any]) -> TSStepResult:
        self.s_base_mva = float(scenario.get("s_base_mva", 100.0))
        p_mw = float(scenario.get("datacenter_rated_mw", 150.0))
        pf = float(scenario.get("datacenter_pf", 0.98))
        q_mvar = p_mw * math.tan(math.acos(np.clip(pf, 0.5, 1.0)))
        self.p_load_pu = p_mw / self.s_base_mva
        self.q_load_pu = q_mvar / self.s_base_mva

        self.v_init_pu = float(scenario.get("v_init_pu", 1.00559))
        self.v_init_deg = float(scenario.get("v_init_deg", -72.489))
        self.v_mag = self.v_init_pu
        self.v_ang_rad = math.radians(self.v_init_deg)

        dist = scenario.get("disturbance", {})
        self.dist_type = dist.get("type", "three_phase_fault")
        self.t_start = float(dist.get("t_start_s", 1.0))
        self.t_clear = float(dist.get("t_clear_s", 1.0833))
        z_f = max(float(dist.get("fault_impedance_pu", 0.30)), 0.02)
        self.g_fault_on = (1.0 / z_f) if self.dist_type in ("three_phase_fault", "voltage_sag") else 0.0
        self.load_step_pu = float(dist.get("load_step_mw", 0.0)) / self.s_base_mva

        # Compute steady-state internal angle & EMF matching pre-disturbance V and P, Q
        p_e0 = self.p_load_pu
        q_e0 = self.q_load_pu + (self.x_eq * (p_e0 ** 2 + self.q_load_pu ** 2)) / (self.v_mag ** 2)
        v_re = self.v_mag
        e_re = v_re + self.x_eq * q_e0 / v_re
        e_im = self.x_eq * p_e0 / v_re
        eq0 = math.hypot(e_re, e_im)
        delta0 = self.v_ang_rad + math.atan2(e_im, e_re)
        self.v_ref = self.v_mag + eq0 / self.k_avr
        self.x = np.array([delta0, 1.0, eq0, p_e0], dtype=np.float64)

        return TSStepResult(
            t_s=0.0,
            v_mag_pu=self.v_mag,
            v_angle_deg=self.v_init_deg,
            freq_hz=60.0,
            min_bus_v_pu=self.v_mag,
            max_rotor_spread_rad=0.0,
            converged=True,
            event_active=False,
        )

    def apply_boundary_load(self, p_mw: float, q_mvar: float) -> None:
        self.p_load_pu = max(0.0, p_mw) / self.s_base_mva
        self.q_load_pu = q_mvar / self.s_base_mva

    def _event_shunt_and_step(self, t: float) -> tuple[float, float, bool]:
        if self.dist_type in ("three_phase_fault", "voltage_sag"):
            on = self.t_start <= t < self.t_clear
            return (self.g_fault_on if on else 0.0), 0.0, on
        if self.dist_type == "load_step":
            on = t >= self.t_start
            return 0.0, (self.load_step_pu if on else 0.0), on
        return 0.0, 0.0, False

    def _solve_bus_algebraic(self, x: np.ndarray, t: float) -> tuple[float, float, float]:
        """Solve boundary bus power balance for (V_mag, V_ang_rad, P_e)."""
        delta, _, e_q, _ = x
        g_fault, dp_extra, _ = self._event_shunt_and_step(t)
        p_target = self.p_load_pu + dp_extra
        q_target = self.q_load_pu

        v = max(self.v_mag, 0.25)
        ang = self.v_ang_rad
        for _ in range(15):
            sin_d = math.sin(delta - ang)
            cos_d = math.cos(delta - ang)
            p_gen = (e_q * v / self.x_eq) * sin_d
            q_gen = (e_q * v * cos_d - v * v) / self.x_eq
            f1 = p_gen - (p_target + g_fault * v * v)
            f2 = q_gen - q_target
            if max(abs(f1), abs(f2)) < 1e-9:
                break
            # Exact 2x2 Jacobian d[f1,f2]/d[v, ang]
            j11 = (e_q / self.x_eq) * sin_d - 2.0 * g_fault * v
            j12 = -(e_q * v / self.x_eq) * cos_d
            j21 = (e_q * cos_d - 2.0 * v) / self.x_eq
            j22 = (e_q * v / self.x_eq) * sin_d
            det = j11 * j22 - j12 * j21
            if abs(det) < 1e-9:
                break
            dv = -(j22 * f1 - j12 * f2) / det
            da = -(-j21 * f1 + j11 * f2) / det
            v = float(np.clip(v + dv, 0.15, 1.35))
            ang = float(ang + np.clip(da, -0.3, 0.3))
        p_e = (e_q * v / self.x_eq) * math.sin(delta - ang)
        return v, ang, p_e

    def _eval_ode(self, x: np.ndarray, t: float) -> tuple[np.ndarray, float, float]:
        _, omega, e_q, p_m = x
        v, ang, p_e = self._solve_bus_algebraic(x, t)
        d_delta = self.omega_0 * (omega - 1.0)
        d_omega = (p_m - p_e - self.d_damp * (omega - 1.0)) / (2.0 * self.h_const)
        d_eq = (self.k_avr * (self.v_ref - v) - e_q) / self.tau_avr
        d_pm = (self.p_load_pu - (omega - 1.0) / self.r_droop - p_m) / self.tau_gov
        return np.array([d_delta, d_omega, d_eq, d_pm], dtype=np.float64), v, ang

    def step(self, t_target: float, dt: float) -> TSStepResult:
        t_prev = t_target - dt
        f_prev, _, _ = self._eval_ode(self.x, t_prev)
        x_next = self.x.copy()
        v_sol, ang_sol = self.v_mag, self.v_ang_rad

        converged = False
        for _ in range(20):
            f_cur, v_sol, ang_sol = self._eval_ode(x_next, t_target)
            res = x_next - self.x - 0.5 * dt * (f_prev + f_cur)
            if np.max(np.abs(res)) < 1e-8:
                converged = True
                break
            # Finite-difference 4x4 Jacobian of implicit trapezoidal residual
            jac = np.eye(4)
            eps = 1e-6
            for j in range(4):
                xp = x_next.copy()
                xp[j] += eps
                fp, _, _ = self._eval_ode(xp, t_target)
                jac[:, j] -= 0.5 * dt * (fp - f_cur) / eps
            dx = np.linalg.solve(jac, -res)
            x_next += dx

        self.x = x_next
        self.v_mag = v_sol
        self.v_ang_rad = ang_sol
        _, _, active = self._event_shunt_and_step(t_target)
        return TSStepResult(
            t_s=t_target,
            v_mag_pu=v_sol,
            v_angle_deg=math.degrees(ang_sol),
            freq_hz=60.0 * float(self.x[1]),
            min_bus_v_pu=v_sol,
            max_rotor_spread_rad=abs(float(self.x[0]) - ang_sol),
            converged=converged,
            event_active=active,
        )

    def get_event_times(self) -> List[float]:
        if self.dist_type in ("three_phase_fault", "voltage_sag"):
            return [self.t_start, self.t_clear]
        if self.dist_type == "load_step":
            return [self.t_start]
        return []


def create_ts_backend(name: str, **kwargs) -> TSBackend:
    """Factory returning the requested TS solver backend."""
    key = name.lower().strip()
    if key == "tsaopt":
        return TSAOptBackend(**kwargs)
    if key == "psse":
        return PSSEBackend()
    if key in ("synthetic", "test"):
        return SyntheticGridBackend()
    raise ValueError(f"Unknown TS backend '{name}'. Expected: tsaopt, psse, synthetic.")
