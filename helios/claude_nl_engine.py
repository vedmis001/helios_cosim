"""
claude_nl_engine.py
═══════════════════════════════════════════════════════════════════════════════
Natural-Language -> Claude API -> Structured JSON -> HELICS Co-Sim -> Summary

Uses the Anthropic Messages API (`ANTHROPIC_API_KEY`) with strict tool-use JSON
schema to convert natural-language power-grid co-simulation prompts into a
validated scenario configuration JSON, run the adaptive co-simulation, and
summarize the output metrics.

Includes `--local-parse` fallback when running offline without an API key.
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, Optional
import urllib.error
import urllib.request

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_CLAUDE_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

SCENARIO_TOOL_SCHEMA: Dict[str, Any] = {
    "name": "configure_cosim_scenario",
    "description": (
        "Produce a validated JSON configuration for the HELICS EMT/TS "
        "co-simulation with learned adaptive stepping and optional TSAOpt backend."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "scenario_name": {"type": "string"},
            "backend": {
                "type": "string",
                "enum": ["synthetic", "tsaopt", "psse"],
                "description": "Transient stability solver backend.",
            },
            "testcase": {
                "type": "string",
                "description": "Optional path to .xlsx testcase when backend='tsaopt' (e.g., testcases/ieee14_loadstep.xlsx).",
            },
            "use_tx_stepping": {"type": "boolean"},
            "sim_time_s": {"type": "number", "minimum": 0.2, "maximum": 60.0},
            "boundary_bus": {"type": "string"},
            "datacenter_rated_mw": {"type": "number", "minimum": 1.0, "maximum": 1000.0},
            "datacenter_pf": {"type": "number", "minimum": 0.7, "maximum": 1.0},
            "component_profile": {
                "type": "string",
                "enum": ["datacenter_vfd", "inverter_gfl", "constant_power"],
            },
            "disturbance": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["three_phase_fault", "voltage_sag", "load_step", "none"],
                    },
                    "bus": {"type": "string"},
                    "t_start_s": {"type": "number", "minimum": 0.0},
                    "t_clear_s": {"type": "number", "minimum": 0.0},
                    "fault_impedance_pu": {"type": "number", "minimum": 0.01, "maximum": 5.0},
                    "load_step_mw": {"type": "number"},
                },
                "required": ["type", "t_start_s"],
            },
            "scheduler": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["learned_adaptive", "fixed"],
                    },
                    "dt_min_s": {"type": "number", "minimum": 0.0005, "maximum": 0.02},
                    "dt_max_s": {"type": "number", "minimum": 0.01, "maximum": 0.25},
                    "dt_init_s": {"type": "number", "minimum": 0.001, "maximum": 0.05},
                    "surprise_tol_mw": {"type": "number", "minimum": 0.2, "maximum": 25.0},
                    "uncertainty_tol": {"type": "number", "minimum": 0.02, "maximum": 1.0},
                    "allow_surrogate_bypass": {"type": "boolean"},
                },
                "required": ["mode"],
            },
        },
        "required": [
            "scenario_name",
            "backend",
            "sim_time_s",
            "boundary_bus",
            "datacenter_rated_mw",
            "component_profile",
            "disturbance",
            "scheduler",
        ],
    },
}

SYSTEM_PROMPT = (
    "You are the HELIOS co-simulation scenario compiler. Translate the user's "
    "natural-language request into exact parameters for `configure_cosim_scenario`. "
    "Defaults when unspecified: backend='synthetic', sim_time_s=4.0, boundary_bus='5385', "
    "datacenter_rated_mw=150.0, datacenter_pf=0.98, component_profile='datacenter_vfd', "
    "disturbance.t_start_s=1.0, 5 cycles at 60Hz = 0.0833s clearing duration, "
    "scheduler.mode='learned_adaptive', dt_min_s=0.002, dt_max_s=0.08, "
    "allow_surrogate_bypass=true."
)


def call_claude_messages(
    api_key: str,
    payload: Dict[str, Any],
    timeout_s: int = 45,
) -> Dict[str, Any]:
    """Send a request to the Anthropic Messages API using stdlib urllib."""
    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Claude API HTTP {exc.code}: {body}") from exc


def nl_to_scenario_claude(
    prompt: str,
    api_key: str,
    model: str = DEFAULT_CLAUDE_MODEL,
) -> Dict[str, Any]:
    """Use Claude tool-use to compile natural language into validated scenario JSON."""
    payload = {
        "model": model,
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "tools": [SCENARIO_TOOL_SCHEMA],
        "tool_choice": {"type": "tool", "name": "configure_cosim_scenario"},
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = call_claude_messages(api_key, payload)
    for block in resp.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "configure_cosim_scenario":
            return validate_scenario(block["input"])
    raise RuntimeError(f"Claude response did not contain tool_use output: {resp}")


def nl_to_scenario_local(prompt: str) -> Dict[str, Any]:
    """Deterministic offline NL parser fallback for local testing without API key."""
    text = prompt.lower()

    backend = "synthetic"
    if "tsaopt" in text:
        backend = "tsaopt"
    elif "psse" in text or "pss/e" in text:
        backend = "psse"

    profile = "datacenter_vfd"
    if "inverter" in text or "gfl" in text:
        profile = "inverter_gfl"
    elif "constant power" in text:
        profile = "constant_power"

    dist_type = "three_phase_fault"
    if "load step" in text or "step load" in text:
        dist_type = "load_step"
    elif "sag" in text:
        dist_type = "voltage_sag"
    elif "no fault" in text or "flat" in text:
        dist_type = "none"

    sim_m = re.search(r"(\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?\b", text)
    sim_time = float(sim_m.group(1)) if sim_m else 4.0
    sim_time = max(0.5, min(sim_time, 30.0))

    mw_m = re.search(r"(\d+(?:\.\d+)?)\s*mw\b", text)
    rated_mw = float(mw_m.group(1)) if mw_m else 150.0

    cyc_m = re.search(r"(\d+(?:\.\d+)?)\s*cycles?\b", text)
    clear_dur = (float(cyc_m.group(1)) / 60.0) if cyc_m else 0.0833

    bus_m = re.search(r"bus\s*(\d+)", text)
    bus = bus_m.group(1) if bus_m else "5385"

    mode = "fixed" if ("fixed" in text and "adaptive" not in text) else "learned_adaptive"
    allow_bypass = "no bypass" not in text and "no surrogate" not in text

    return validate_scenario(
        {
            "scenario_name": "nl_parsed_scenario",
            "backend": backend,
            "sim_time_s": sim_time,
            "boundary_bus": bus,
            "datacenter_rated_mw": rated_mw,
            "datacenter_pf": 0.98,
            "component_profile": profile,
            "disturbance": {
                "type": dist_type,
                "bus": bus,
                "t_start_s": 0.5 if sim_time <= 2.5 else 1.0,
                "t_clear_s": (0.5 if sim_time <= 2.5 else 1.0) + clear_dur,
                "fault_impedance_pu": 0.25 if "severe" in text or "bolted" in text else 0.35,
                "load_step_mw": 40.0 if dist_type == "load_step" else 0.0,
            },
            "scheduler": {
                "mode": mode,
                "dt_min_s": 0.002,
                "dt_max_s": 0.08,
                "dt_init_s": 0.01,
                "surprise_tol_mw": 3.5,
                "uncertainty_tol": 0.18,
                "allow_surrogate_bypass": allow_bypass,
            },
        }
    )


def validate_scenario(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply physical bounds and defaults to a scenario dictionary."""
    cfg.setdefault("datacenter_pf", 0.98)
    cfg.setdefault("s_base_mva", 100.0)
    dist = cfg.setdefault("disturbance", {})
    dist.setdefault("bus", cfg.get("boundary_bus", "5385"))
    t_start = float(dist.get("t_start_s", 1.0))
    t_clear = float(dist.get("t_clear_s", t_start + 0.0833))
    if t_clear <= t_start:
        t_clear = t_start + 0.0833
    dist["t_start_s"] = t_start
    dist["t_clear_s"] = t_clear
    dist.setdefault("fault_impedance_pu", 0.35)
    dist.setdefault("load_step_mw", 0.0)

    sched = cfg.setdefault("scheduler", {})
    sched.setdefault("mode", "learned_adaptive")
    sched.setdefault("dt_min_s", 0.002)
    sched.setdefault("dt_max_s", 0.08)
    sched.setdefault("dt_init_s", 0.01)
    sched.setdefault("surprise_tol_mw", 3.5)
    sched.setdefault("uncertainty_tol", 0.18)
    sched.setdefault("allow_surrogate_bypass", True)
    return cfg


def summarize_results_claude(
    prompt: str,
    scenario: Dict[str, Any],
    results: Dict[str, Any],
    api_key: str,
    model: str = DEFAULT_CLAUDE_MODEL,
) -> str:
    """Have Claude summarize simulation KPIs in plain engineering language."""
    payload = {
        "model": model,
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Original request: {prompt}\n\n"
                    f"Scenario JSON:\n{json.dumps(scenario, indent=2)}\n\n"
                    f"Simulation Output JSON:\n{json.dumps(results, indent=2)}\n\n"
                    "Summarize stability, boundary voltage/frequency excursions, "
                    "and adaptive stepping / surrogate efficiency concisely."
                ),
            }
        ],
    }
    resp = call_claude_messages(api_key, payload)
    texts = [b["text"] for b in resp.get("content", []) if b.get("type") == "text"]
    return "\n".join(texts).strip()


def main():
    parser = argparse.ArgumentParser(description="Claude NL -> JSON -> HELICS Co-Sim Engine")
    parser.add_argument("prompt", help="Natural language simulation request")
    parser.add_argument("--dry-run", action="store_true", help="Only emit compiled scenario JSON")
    parser.add_argument("--local-parse", action="store_true", help="Force offline parser instead of Claude API")
    parser.add_argument("--core", default="inproc", choices=["inproc", "zmq"])
    parser.add_argument("--weights", default="learned_weights.json")
    parser.add_argument("--out-scenario", default="nl_scenario.json")
    parser.add_argument("--out-results", default="helios_results.json")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not args.local_parse and api_key:
        scenario = nl_to_scenario_claude(args.prompt, api_key)
        used_api = True
    else:
        if not args.local_parse and not api_key:
            print("[INFO] ANTHROPIC_API_KEY not set; using offline --local-parse fallback.", file=sys.stderr)
        scenario = nl_to_scenario_local(args.prompt)
        used_api = False

    Path(args.out_scenario).write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    print("=== Compiled Scenario JSON ===")
    print(json.dumps(scenario, indent=2))

    if args.dry_run:
        return

    from run_adaptive_cosim import run_cosim_pair
    results = run_cosim_pair(
        scenario=scenario,
        core_type=args.core,
        weights_path=args.weights,
        results_path=args.out_results,
    )
    print("\n=== Simulation Output JSON ===")
    print(json.dumps(results, indent=2))

    if used_api and api_key:
        print("\n=== Claude Engineering Summary ===")
        print(summarize_results_claude(args.prompt, scenario, results, api_key))


if __name__ == "__main__":
    main()
