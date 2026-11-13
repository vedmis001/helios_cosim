"""
helios_ts_federate.py
═══════════════════════════════════════════════════════════════════════════════
HELIOS Co-Simulation — Adaptive Transient Stability (TS) Federate

Integrates:
  - Pluggable TS backend (`tsaopt`, `psse`, or `synthetic` trapezoidal DAE)
  - Online RLS boundary surrogate + epistemic uncertainty gating (`BoundarySurrogateRLS`)
  - Learned derivative-free variable timestep policy (`AdaptiveStepPolicy`)
  - Synchronized HELICS publications (`boundary_V_mag`, `boundary_V_angle`,
    `next_dt`, `skip_emt`)
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import helics as h
import numpy as np

from learned_scheduler import (
    AdaptiveStepPolicy,
    BoundarySurrogateRLS,
    ScenarioContext,
    SchedulerConfig,
    load_learned_state,
    save_learned_state,
)
from ts_backend import create_ts_backend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TS-FED] %(message)s",
    handlers=[
        logging.FileHandler("helios_ts.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

COMPONENT_ID_MAP = {
    "datacenter_vfd": 0.0,
    "inverter_gfl": 1.0,
    "constant_power": -1.0,
}


def create_ts_federate(
    core_type: str = "zmq",
    broker_name: Optional[str] = None,
    adaptive: bool = True,
    dt_init: float = 10e-3,
    rated_p_mw: float = 150.0,
    rated_q_mvar: float = 30.6,
):
    """Create and configure the TS HELICS value federate."""
    fedinfo = h.helicsCreateFederateInfo()
    h.helicsFederateInfoSetCoreName(fedinfo, f"psse_core_{id(fedinfo)}")
    h.helicsFederateInfoSetCoreTypeFromString(fedinfo, core_type)
    init_str = "--federates=1" + (f" --broker={broker_name}" if broker_name else "")
    h.helicsFederateInfoSetCoreInitString(fedinfo, init_str)

    period = 0.0 if adaptive else dt_init
    h.helicsFederateInfoSetTimeProperty(fedinfo, h.HELICS_PROPERTY_TIME_PERIOD, period)
    h.helicsFederateInfoSetFlagOption(fedinfo, h.HELICS_FLAG_SLOW_RESPONDING, True)
    h.helicsFederateInfoSetFlagOption(fedinfo, h.HELICS_FLAG_UNINTERRUPTIBLE, True)

    fed = h.helicsCreateValueFederate("psse_federate", fedinfo)

    pub_V_mag = h.helicsFederateRegisterGlobalPublication(
        fed, "psse_federate/boundary_V_mag", h.HELICS_DATA_TYPE_DOUBLE, "pu"
    )
    pub_V_angle = h.helicsFederateRegisterGlobalPublication(
        fed, "psse_federate/boundary_V_angle", h.HELICS_DATA_TYPE_DOUBLE, "deg"
    )
    pub_next_dt = h.helicsFederateRegisterGlobalPublication(
        fed, "psse_federate/next_dt", h.HELICS_DATA_TYPE_DOUBLE, "s"
    )
    pub_skip_emt = h.helicsFederateRegisterGlobalPublication(
        fed, "psse_federate/skip_emt", h.HELICS_DATA_TYPE_BOOLEAN, ""
    )

    sub_P = h.helicsFederateRegisterSubscription(fed, "pscad_federate/P_injection", "MW")
    sub_Q = h.helicsFederateRegisterSubscription(fed, "pscad_federate/Q_injection", "MVAR")
    h.helicsInputSetDefaultDouble(sub_P, rated_p_mw)
    h.helicsInputSetDefaultDouble(sub_Q, rated_q_mvar)

    return fed, pub_V_mag, pub_V_angle, pub_next_dt, pub_skip_emt, sub_P, sub_Q


def run_ts_federate(
    scenario: Dict[str, Any],
    core_type: str = "zmq",
    broker_name: Optional[str] = None,
    weights_path: Optional[str] = None,
    results_path: Optional[str] = None,
    close_library: bool = True,
    backend_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the TS HELICS federate with learned adaptive stepping."""
    t_wall_start = time.perf_counter()

    backend_name = scenario.get("backend", "synthetic")
    backend = create_ts_backend(backend_name, **(backend_kwargs or {}))
    init_res = backend.initialize(scenario)

    total_time = float(scenario.get("sim_time_s", 4.0))
    rated_p = float(scenario.get("datacenter_rated_mw", 150.0))
    pf = float(scenario.get("datacenter_pf", 0.98))
    rated_q = rated_p * math.tan(math.acos(min(1.0, max(0.5, pf))))

    sched_dict = scenario.get("scheduler", {})
    sched_cfg = SchedulerConfig(
        mode=sched_dict.get("mode", "learned_adaptive"),
        dt_fixed_s=float(sched_dict.get("dt_init_s", 10e-3)),
        dt_min_s=float(sched_dict.get("dt_min_s", 2e-3)),
        dt_max_s=float(sched_dict.get("dt_max_s", 80e-3)),
        dt_init_s=float(sched_dict.get("dt_init_s", 10e-3)),
        surprise_tol_mw=float(sched_dict.get("surprise_tol_mw", 3.5)),
        uncertainty_tol=float(sched_dict.get("uncertainty_tol", 0.18)),
        allow_surrogate_bypass=bool(sched_dict.get("allow_surrogate_bypass", True)),
    )
    adaptive = sched_cfg.mode == "learned_adaptive"

    dist = scenario.get("disturbance", {})
    z_fault = float(dist.get("fault_impedance_pu", 0.35))
    ctx = ScenarioContext(
        rated_p_mw=rated_p,
        rated_q_mvar=rated_q,
        v_init_pu=init_res.v_mag_pu,
        fault_severity=(1.0 / max(z_fault, 0.05)) if dist.get("type", "none") != "none" else 0.0,
        component_id=COMPONENT_ID_MAP.get(scenario.get("component_profile", "datacenter_vfd"), 0.0),
        event_times=backend.get_event_times(),
    )

    if weights_path:
        surrogate, policy = load_learned_state(weights_path, sched_cfg, ctx)
    else:
        surrogate = BoundarySurrogateRLS(p_init=rated_p, q_init=rated_q)
        policy = AdaptiveStepPolicy(sched_cfg, ctx)

    fed, pub_V_mag, pub_V_angle, pub_next_dt, pub_skip_emt, sub_P, sub_Q = create_ts_federate(
        core_type=core_type,
        broker_name=broker_name,
        adaptive=adaptive,
        dt_init=sched_cfg.dt_init_s,
        rated_p_mw=rated_p,
        rated_q_mvar=rated_q,
    )

    dt_curr = sched_cfg.dt_init_s
    dt_sched = sched_cfg.dt_init_s
    skip_curr = False
    skip_sched = False

    h.helicsFederateEnterInitializingMode(fed)
    h.helicsPublicationPublishDouble(pub_V_mag, init_res.v_mag_pu)
    h.helicsPublicationPublishDouble(pub_V_angle, init_res.v_angle_deg)
    h.helicsPublicationPublishDouble(pub_next_dt, dt_sched)
    h.helicsPublicationPublishBoolean(pub_skip_emt, skip_sched)
    h.helicsFederateEnterExecutingMode(fed)

    current_time = 0.0
    step = 0
    emt_sync_count = 0
    surrogate_bypass_count = 0

    prev_res = init_res
    p_curr, q_curr = rated_p, rated_q
    prev_phi = BoundarySurrogateRLS.build_features(
        v_mag=init_res.v_mag_pu,
        dv=0.0,
        v_ang_rad=math.radians(init_res.v_angle_deg),
        dangle_rad=0.0,
        p_prev=p_curr,
        q_prev=q_curr,
        dt=dt_curr,
        ctx=ctx,
    )

    trace_t: List[float] = [0.0]
    trace_v: List[float] = [init_res.v_mag_pu]
    trace_f: List[float] = [init_res.freq_hz]
    trace_p: List[float] = [rated_p]
    trace_dt: List[float] = []
    surprise_history: List[float] = []

    while current_time < total_time - 1e-9:
        target_t = min(total_time, current_time + dt_curr)
        granted_time = h.helicsFederateRequestTime(fed, target_t)
        dt_actual = max(1e-6, granted_time - current_time)
        step += 1

        if skip_curr:
            surrogate_bypass_count += 1
            p_apply, q_apply, unc = surrogate.predict(prev_phi)
            surprise_mw = 0.0
        else:
            emt_sync_count += 1
            p_meas = h.helicsInputGetDouble(sub_P)
            q_meas = h.helicsInputGetDouble(sub_Q)
            err_p, _ = surrogate.update(prev_phi, p_meas, q_meas)
            surprise_mw = err_p
            surprise_history.append(err_p)
            p_apply, q_apply = p_meas, q_meas
            _, _, unc = surrogate.predict(prev_phi)

        backend.apply_boundary_load(p_apply, q_apply)
        res = backend.step(granted_time, dt_actual)

        dv = res.v_mag_pu - prev_res.v_mag_pu
        dangle = math.radians(res.v_angle_deg - prev_res.v_angle_deg)
        df_hz = res.freq_hz - 60.0

        phi_next = BoundarySurrogateRLS.build_features(
            v_mag=res.v_mag_pu,
            dv=dv,
            v_ang_rad=math.radians(res.v_angle_deg),
            dangle_rad=dangle,
            p_prev=p_apply,
            q_prev=q_apply,
            dt=dt_sched,
            ctx=ctx,
        )
        _, _, unc_next = surrogate.predict(phi_next)

        # Schedule the step starting at (granted_time + dt_sched) so PSCAD stays synchronized
        t_exec_future = granted_time + dt_sched
        dt_next_sched, skip_next_sched, _ = policy.select_action(
            t_now=t_exec_future,
            t_end=total_time,
            surprise_mw=surprise_mw,
            uncertainty=unc_next,
            dv_pu=dv,
            df_hz=df_hz,
        )

        # Advance 1-step-ahead schedule pipeline
        dt_curr = dt_sched
        skip_curr = skip_sched
        dt_sched = dt_next_sched
        skip_sched = skip_next_sched

        h.helicsPublicationPublishDouble(pub_V_mag, res.v_mag_pu)
        h.helicsPublicationPublishDouble(pub_V_angle, res.v_angle_deg)
        h.helicsPublicationPublishDouble(pub_next_dt, dt_sched)
        h.helicsPublicationPublishBoolean(pub_skip_emt, skip_sched)

        trace_t.append(round(granted_time, 6))
        trace_v.append(round(res.v_mag_pu, 6))
        trace_f.append(round(res.freq_hz, 6))
        trace_p.append(round(p_apply, 4))
        trace_dt.append(round(dt_actual, 6))

        prev_res = res
        prev_phi = phi_next
        p_curr, q_curr = p_apply, q_apply
        current_time = granted_time

    backend.finalize()
    h.helicsFederateDisconnect(fed)
    h.helicsFederateFree(fed)
    if close_library:
        h.helicsCloseLibrary()

    if weights_path:
        save_learned_state(weights_path, surrogate, policy)

    wall_s = time.perf_counter() - t_wall_start
    fixed_ref_steps = int(round(total_time / sched_cfg.dt_fixed_s))

    results = {
        "scenario_name": scenario.get("scenario_name", "unnamed"),
        "backend": backend_name,
        "scheduler_mode": sched_cfg.mode,
        "component_profile": scenario.get("component_profile", "datacenter_vfd"),
        "sim_time_s": total_time,
        "wall_time_s": round(wall_s, 4),
        "total_steps": step,
        "fixed_10ms_baseline_steps": fixed_ref_steps,
        "emt_sync_calls": emt_sync_count,
        "surrogate_bypasses": surrogate_bypass_count,
        "emt_reduction_pct": round(100.0 * (1.0 - emt_sync_count / max(fixed_ref_steps, 1)), 2),
        "avg_dt_ms": round(1000.0 * float(np.mean(trace_dt)), 3) if trace_dt else 0.0,
        "min_dt_ms": round(1000.0 * float(np.min(trace_dt)), 3) if trace_dt else 0.0,
        "max_dt_ms": round(1000.0 * float(np.max(trace_dt)), 3) if trace_dt else 0.0,
        "min_boundary_v_pu": round(float(np.min(trace_v)), 5),
        "max_boundary_v_pu": round(float(np.max(trace_v)), 5),
        "freq_nadir_hz": round(float(np.min(trace_f)), 5),
        "freq_peak_hz": round(float(np.max(trace_f)), 5),
        "mean_surrogate_err_mw": round(float(np.mean(surprise_history)), 4) if surprise_history else 0.0,
        "rls_updates": surrogate.n_updates,
        "trace_sample": {
            "t": trace_t[::max(1, len(trace_t) // 25)],
            "v_pu": trace_v[::max(1, len(trace_v) // 25)],
            "freq_hz": trace_f[::max(1, len(trace_f) // 25)],
            "p_mw": trace_p[::max(1, len(trace_p) // 25)],
        },
    }

    if results_path:
        Path(results_path).write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def main():
    parser = argparse.ArgumentParser(description="HELIOS Adaptive TS Federate")
    parser.add_argument("--config", required=True, help="Path to scenario JSON config")
    parser.add_argument("--core", default="zmq", choices=["zmq", "inproc"])
    parser.add_argument("--weights", default="learned_weights.json")
    parser.add_argument("--out", default="helios_results.json")
    args = parser.parse_args()

    scenario = json.loads(Path(args.config).read_text(encoding="utf-8"))
    res = run_ts_federate(
        scenario=scenario,
        core_type=args.core,
        weights_path=args.weights,
        results_path=args.out,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
