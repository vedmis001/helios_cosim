"""
run_adaptive_cosim.py
═══════════════════════════════════════════════════════════════════════════════
Single-command HELICS co-simulation launcher (inproc or zmq) pairing
`helios_ts_federate` and `helios_pscad_federate` with learned adaptive stepping.
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import math
from pathlib import Path
import threading
from typing import Any, Dict, Optional
import helics as h

from helios_pscad_federate import run_cosim as run_pscad_loop
from helios_ts_federate import run_ts_federate


def run_cosim_pair(
    scenario: Dict[str, Any],
    core_type: str = "inproc",
    weights_path: Optional[str] = None,
    results_path: Optional[str] = None,
    simulate_delay: bool = True,
) -> Dict[str, Any]:
    """Run broker + TS federate + PSCAD federate together and return metrics."""
    broker_name = f"helios_broker_{id(scenario)}"
    broker = h.helicsCreateBroker(core_type, broker_name, "--federates=2")

    sched = scenario.get("scheduler", {})
    adaptive = sched.get("mode", "learned_adaptive") == "learned_adaptive"
    total_time = float(scenario.get("sim_time_s", 4.0))
    dt_init = float(sched.get("dt_init_s", 10e-3))
    rated_p = float(scenario.get("datacenter_rated_mw", 150.0))
    pf = float(scenario.get("datacenter_pf", 0.98))
    rated_q = rated_p * math.tan(math.acos(min(1.0, max(0.5, pf))))
    profile = scenario.get("component_profile", "datacenter_vfd")

    ts_out: Dict[str, Any] = {}
    errors = []

    def _ts_worker():
        try:
            ts_out["res"] = run_ts_federate(
                scenario=scenario,
                core_type=core_type,
                broker_name=broker_name,
                weights_path=weights_path,
                results_path=results_path,
                close_library=False,
            )
        except Exception as exc:
            errors.append(("ts", exc))

    def _pscad_worker():
        try:
            ts_out["pscad"] = run_pscad_loop(
                total_time=total_time,
                exchange_interval=dt_init,
                adaptive=adaptive,
                core_type=core_type,
                broker_name=broker_name,
                rated_p_mw=rated_p,
                rated_q_mvar=rated_q,
                profile=profile,
                simulate_delay=simulate_delay,
                close_library=False,
            )
        except Exception as exc:
            errors.append(("pscad", exc))

    t_ts = threading.Thread(target=_ts_worker, daemon=True)
    t_pscad = threading.Thread(target=_pscad_worker, daemon=True)
    t_ts.start()
    t_pscad.start()
    t_ts.join()
    t_pscad.join()

    h.helicsBrokerDisconnect(broker)
    h.helicsBrokerFree(broker)
    h.helicsCloseLibrary()

    if errors:
        raise RuntimeError(f"Federate failure: {errors}")

    res = ts_out["res"]
    res["pscad_stats"] = ts_out["pscad"]
    return res


def main():
    parser = argparse.ArgumentParser(description="Run HELICS Adaptive Co-Simulation")
    parser.add_argument("--config", required=True, help="Scenario JSON path")
    parser.add_argument("--core", default="inproc", choices=["inproc", "zmq"])
    parser.add_argument("--weights", default="learned_weights.json")
    parser.add_argument("--out", default="helios_results.json")
    parser.add_argument("--no-delay", action="store_true", help="Disable EMT latency proxy")
    args = parser.parse_args()

    scenario = json.loads(Path(args.config).read_text(encoding="utf-8"))
    res = run_cosim_pair(
        scenario=scenario,
        core_type=args.core,
        weights_path=args.weights,
        results_path=args.out,
        simulate_delay=not args.no_delay,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
