"""
helios_psse_federate.py
═══════════════════════════════════════════════════════════════════════════════
HELIOS Co-Simulation — PSS/E Federate
ACTIVSg2000 Texas Grid (Transient Stability Side)

Boundary Bus:  5385  DALLAS 3 1  161.0 kV
Pre-disturbance: V = 1.00559 pu,  angle = -72.489 deg
Datacenter load: 150 MW, 30.6 MVAR (PF = 0.98), Load ID = 'DC'

Interface convention (standard EMT/TS Norton boundary):
  PUBLISH  → PSCAD : boundary bus V_magnitude (pu), V_angle (deg)
  SUBSCRIBE← PSCAD : datacenter P_injection (MW), Q_injection (MVAR)

Exchange interval: 10ms  (PSS/E TS timestep)
PSCAD runs freely at 0.5us EMT timestep between exchanges

Authors: Vedant Misra, Department of Electrical and Computer Engineering, Carnegie Mellon University
Advised by Professor Larry Pileggi and Dr.Aayushya Agrawal
═══════════════════════════════════════════════════════════════════════════════
"""

import helics as h
import sys
import os
import time

# ─── PSS/E Path Configuration ────────────────────────────────────────────────
PSSE_BIN_PATH = os.environ.get("PSSE_BIN_PATH", r"C:\Program Files\PTI\PSSE36\36.7\PSSBIN")
PSSE_PY_PATH  = os.environ.get("PSSE_PY_PATH",  r"C:\Program Files\PTI\PSSE36\36.7\PSSPY313")

if os.path.exists(PSSE_PY_PATH) and PSSE_PY_PATH not in sys.path:
    sys.path.insert(0, PSSE_PY_PATH)
if os.path.exists(PSSE_BIN_PATH) and PSSE_BIN_PATH not in sys.path:
    sys.path.insert(0, PSSE_BIN_PATH)
if os.path.exists(PSSE_BIN_PATH):
    os.environ["PATH"] = PSSE_BIN_PATH + ";" + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(PSSE_BIN_PATH)
        except Exception:
            pass

import psspy
import dyntools
import logging
import math
import numpy as np

from learned_scheduler import (
    BoundarySurrogateRLS,
    AdaptiveStepPolicy,
    SchedulerConfig,
    ScenarioContext,
)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s [PSSE] %(message)s",
    handlers = [
        logging.FileHandler("helios_psse.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

# File paths -- update if files are in a different directory
PSSE_SAV_FILE = r"C:\Users\vamisra\HELIOS\psse\ACTIVSg2000_HELIOS_DC150MW.sav"
PSSE_SNP_FILE = r"C:\Users\vamisra\HELIOS\psse\ACTIVSg2000_HELIOS_t0.snp"
PSSE_DYR_FILE = r"C:\Users\vamisra\Documents\ACTIVSg2000_dynamics.dyr"
OUTPUT_FILE   = "helios_psse_channels.out"


# Boundary bus — Bus 5385, DALLAS 3 1, 161kV
BOUNDARY_BUS  = 5385
V_BASE_KV     = 161.0
S_BASE_MVA    = 100.0

# Pre-disturbance operating point from PSS/E power flow
V_INIT_PU     = 1.00559    # pu
V_INIT_DEG    = -72.489    # degrees

# Datacenter load parameters
DC_P_MW       = 150.0
DC_Q_MVAR     = DC_P_MW * math.tan(math.acos(0.98))   # 30.6 MVAR
DC_LOAD_ID    = r"""DC"""

# HELICS timing
EXCHANGE_INTERVAL = float(os.environ.get("HELIOS_EXCHANGE_INTERVAL", "10e-3"))
TOTAL_SIM_TIME    = float(os.environ.get("HELIOS_TOTAL_SIM_TIME", "10.0"))
ADAPTIVE_STEPPING = os.environ.get("HELIOS_ADAPTIVE_STEPPING", "1") == "1"

# ─── PSS/E Initialization ────────────────────────────────────────────────────

def initialize_psse():
    """
    Initialize PSS/E from saved snapshot.
    This is fast -- skips the full psseinit/read/fnsl/cong/strt_2 sequence.
    """
    log.info("Initializing PSS/E...")

    # Initialize PSS/E memory
    psspy.psseinit(buses=80000)

    # Suppress PSS/E output to console during co-simulation
    psspy.progress_output(6, "", [])   # redirect progress to null
    psspy.alert_output(6, "", [])      # redirect alerts to null
    psspy.prompt_output(6, "", [])     # redirect prompts to null

    # Load solved power flow case (with 150MW datacenter already at Bus 5385)
    ierr = psspy.case(PSSE_SAV_FILE)
    if ierr != 0:
        raise RuntimeError(f"Failed to load SAV file: ierr={ierr}")
    log.info(f"  Loaded: {PSSE_SAV_FILE}")

    # Load dynamics data
    ierr = psspy.dyre_new(
        [1,1,1,1], PSSE_DYR_FILE, "", "", ""
    )
    if ierr != 0:
        raise RuntimeError(f"Failed to load DYR file: ierr={ierr}")
    log.info(f"  Loaded: {PSSE_DYR_FILE}")

    # Convert generators (required before restoring snapshot)
    ierr = psspy.cong(0)
    if ierr != 0:
        raise RuntimeError(f"Generator conversion failed: ierr={ierr}")

    # Restore dynamics snapshot -- all states initialized at t=0
    ierr = psspy.rstr(PSSE_SNP_FILE)
    if ierr != 0:
        raise RuntimeError(f"Failed to restore snapshot: ierr={ierr}")
    log.info(f"  Restored snapshot: {PSSE_SNP_FILE}")

    # Verify boundary bus voltage
    ierr, v = psspy.busdat(BOUNDARY_BUS, 'PU')
    log.info(f"  Boundary bus {BOUNDARY_BUS} voltage: {abs(v):.5f} pu "
             f"(expected {V_INIT_PU:.5f} pu)")

    log.info("PSS/E initialized successfully")


def setup_output_channels():
    """Configure PSS/E output channels for monitoring (PSS/E 36 API syntax)."""

    try:
        psspy.delete_all_plot_channels()
    except Exception as e:
        log.warning(f"delete_all_plot_channels warning: {e}")

    # Channel 1: Boundary bus voltage magnitude
    # In PSS/E 36: voltage_channel(status=[-1,-1,-1,bus], ident="...")
    try:
        psspy.voltage_channel([-1, -1, -1, BOUNDARY_BUS], "V_BOUNDARY")
    except Exception as e:
        log.warning(f"Failed to setup V_BOUNDARY channel: {e}")

    # Channel 2: Boundary bus frequency
    # In PSS/E 36: bus_frequency_channel(status=[-1, bus], ident="...")
    try:
        psspy.bus_frequency_channel([-1, BOUNDARY_BUS], "FREQ_BOUNDARY")
    except Exception as e:
        log.warning(f"Failed to setup FREQ_BOUNDARY channel: {e}")

    # Channel 3: System frequency (slack bus 7098)
    try:
        psspy.bus_frequency_channel([-1, 7098], "FREQ_SLACK")
    except Exception as e:
        log.warning(f"Failed to setup FREQ_SLACK channel: {e}")

    # Channel 4: Datacenter active power
    # In PSS/E 36: load_array_channel(status=[-1, 1 (PLOAD), bus], id=DC_LOAD_ID, ident="P_DC")
    try:
        psspy.load_array_channel([-1, 1, BOUNDARY_BUS], DC_LOAD_ID, "P_DC")
    except Exception as e:
        log.warning(f"Failed to setup P_DC channel: {e}")

    # Channel 5: Datacenter reactive power
    # In PSS/E 36: load_array_channel(status=[-1, 2 (QLOAD), bus], id=DC_LOAD_ID, ident="Q_DC")
    try:
        psspy.load_array_channel([-1, 2, BOUNDARY_BUS], DC_LOAD_ID, "Q_DC")
    except Exception as e:
        log.warning(f"Failed to setup Q_DC channel: {e}")

    # Open output file
    try:
        psspy.chsb(0, 1, [-1, -1, -1, 1, 1, 0])
        psspy.change_channel_out_file(OUTPUT_FILE)
        log.info(f"Output channels configured, writing to: {OUTPUT_FILE}")
    except Exception as e:
        log.warning(f"Failed to configure channel output file: {e}")


# ─── Boundary Interface Functions ────────────────────────────────────────────

def get_boundary_voltage():
    """
    Read boundary bus voltage from PSS/E solution.
    Returns (V_magnitude_pu, V_angle_deg).
    """
    ierr_m, v_mag   = psspy.busdat(BOUNDARY_BUS, 'PU')
    ierr_a, v_angle = psspy.busdat(BOUNDARY_BUS, 'ANGLED')

    if ierr_m != 0 or ierr_a != 0:
        log.warning(f"  Boundary voltage read error: "
                    f"mag_err={ierr_m}, ang_err={ierr_a}")
        return V_INIT_PU, V_INIT_DEG   # fall back to initial values

    return abs(v_mag), v_angle


def apply_datacenter_load(P_MW, Q_MVAR):
    """
    Update the datacenter load at the boundary bus with values from PSCAD.
    This replaces the static 150MW load with the dynamic PSCAD output.

    P_MW:   active power injection from PSCAD (MW, positive = load)
    Q_MVAR: reactive power injection from PSCAD (MVAR, positive = inductive)
    """
    # Clamp to physically reasonable bounds
    P_MW   = max(0.0,   min(P_MW,   500.0))    # 0 to 500 MW
    Q_MVAR = max(-100.0, min(Q_MVAR, 200.0))   # -100 to +200 MVAR

    ierr = psspy.load_chng_4(
        BOUNDARY_BUS,
        DC_LOAD_ID,
        [1,1,1,1,1,1],                          # all status flags in-service
        [P_MW, Q_MVAR, 0.0, 0.0, 0.0, 0.0]     # PL, QL, IP, IQ, YP, YQ
    )

    if ierr != 0:
        log.warning(f"  Load update failed at Bus {BOUNDARY_BUS}: ierr={ierr}")

    return ierr


def step_psse(dt):
    """
    Advance PSS/E dynamics by one exchange interval.
    dt: timestep in seconds (= EXCHANGE_INTERVAL = 10ms)
    """
    # run(output, end_time, steps_between_output,
    #     steps_between_network_solution, voltage_monitor)
    ierr = psspy.run(0, dt, 1, 1, 0)
    return ierr


# ─── HELICS Federate Setup ───────────────────────────────────────────────────

def create_psse_federate():
    """Create and configure the PSS/E HELICS value federate."""

    fedinfo = h.helicsCreateFederateInfo()

    # Core configuration
    h.helicsFederateInfoSetCoreName(fedinfo, "psse_core")
    h.helicsFederateInfoSetCoreTypeFromString(fedinfo, "zmq")

    # This federate does NOT start the broker
    h.helicsFederateInfoSetCoreInitString(fedinfo, "--federates=1")

    # Exchange period: 0.0 in adaptive mode allows variable time requests
    period = 0.0 if ADAPTIVE_STEPPING else EXCHANGE_INTERVAL
    h.helicsFederateInfoSetTimeProperty(
        fedinfo,
        h.HELICS_PROPERTY_TIME_PERIOD,
        period
    )

    # PSS/E dynamics can be slow -- prevent broker timeout
    h.helicsFederateInfoSetFlagOption(
        fedinfo,
        h.HELICS_FLAG_SLOW_RESPONDING,
        True
    )

    # Time delta: minimum time between granted times
    time_delta = 1e-4 if ADAPTIVE_STEPPING else EXCHANGE_INTERVAL
    h.helicsFederateInfoSetTimeProperty(
        fedinfo,
        h.HELICS_PROPERTY_TIME_DELTA,
        time_delta
    )

    fed = h.helicsCreateValueFederate("psse_federate", fedinfo)
    log.info("HELICS PSS/E federate created")
    return fed


def register_psse_interfaces(fed):
    """
    Register HELICS publications and subscriptions for PSS/E federate.

    PSS/E PUBLISHES (sends to PSCAD):
      - boundary bus voltage magnitude (pu)
      - boundary bus voltage angle (degrees)
      - next exchange timestep next_dt (s)
      - surrogate bypass flag skip_emt (boolean)

    PSS/E SUBSCRIBES (receives from PSCAD):
      - datacenter active power P (MW)
      - datacenter reactive power Q (MVAR)
    """

    # ── Publications (PSS/E → PSCAD) ─────────────────────────────────────
    pub_V_mag = h.helicsFederateRegisterGlobalPublication(
        fed,
        "psse_federate/boundary_V_mag",
        h.HELICS_DATA_TYPE_DOUBLE,
        "pu"
    )

    pub_V_angle = h.helicsFederateRegisterGlobalPublication(
        fed,
        "psse_federate/boundary_V_angle",
        h.HELICS_DATA_TYPE_DOUBLE,
        "deg"
    )

    pub_next_dt = h.helicsFederateRegisterGlobalPublication(
        fed,
        "psse_federate/next_dt",
        h.HELICS_DATA_TYPE_DOUBLE,
        "s"
    )

    pub_skip_emt = h.helicsFederateRegisterGlobalPublication(
        fed,
        "psse_federate/skip_emt",
        h.HELICS_DATA_TYPE_BOOLEAN,
        ""
    )

    # ── Subscriptions (PSCAD → PSS/E) ────────────────────────────────────
    sub_P = h.helicsFederateRegisterSubscription(
        fed,
        "pscad_federate/P_injection",
        "MW"
    )

    sub_Q = h.helicsFederateRegisterSubscription(
        fed,
        "pscad_federate/Q_injection",
        "MVAR"
    )

    log.info("HELICS interfaces registered:")
    log.info("  Publishing:   boundary_V_mag [pu], boundary_V_angle [deg], next_dt [s], skip_emt [bool]")
    log.info("  Subscribing:  P_injection [MW], Q_injection [MVAR]")

    return pub_V_mag, pub_V_angle, pub_next_dt, pub_skip_emt, sub_P, sub_Q


# ─── Main Co-Simulation Loop ─────────────────────────────────────────────────

def run_cosim():
    """Main entry point for the PSS/E HELICS federate with Adaptive RLS Scheduling."""

    # 1. Initialize PSS/E
    initialize_psse()
    setup_output_channels()

    # 2. Create HELICS federate and register interfaces
    fed = create_psse_federate()
    pub_V_mag, pub_V_angle, pub_next_dt, pub_skip_emt, sub_P, sub_Q = register_psse_interfaces(fed)

    # Adaptive RLS scheduler setup
    sched_cfg = SchedulerConfig(
        mode="learned_adaptive" if ADAPTIVE_STEPPING else "fixed",
        dt_fixed_s=EXCHANGE_INTERVAL,
        dt_min_s=2e-3,
        dt_max_s=80e-3,
        dt_init_s=EXCHANGE_INTERVAL,
        surprise_tol_mw=3.5,
        uncertainty_tol=0.18,
        allow_surrogate_bypass=True,
        max_surrogate_streak=4,
    )
    ctx = ScenarioContext(
        rated_p_mw=DC_P_MW,
        rated_q_mvar=DC_Q_MVAR,
        v_init_pu=V_INIT_PU,
    )
    surrogate = BoundarySurrogateRLS(
        p_init=DC_P_MW,
        q_init=DC_Q_MVAR,
        forgetting_factor=sched_cfg.forgetting_factor,
        prior_reg=sched_cfg.prior_reg,
    )
    policy = AdaptiveStepPolicy(sched_cfg, ctx)

    dt_curr = sched_cfg.dt_init_s
    dt_sched = sched_cfg.dt_init_s
    skip_curr = False
    skip_sched = False

    # 3. Enter initialization mode
    log.info("Entering HELICS initialization mode...")
    h.helicsFederateEnterInitializingMode(fed)

    # Publish initial boundary voltage before execution starts
    V_mag_init, V_ang_init = get_boundary_voltage()
    h.helicsPublicationPublishDouble(pub_V_mag,   V_mag_init)
    h.helicsPublicationPublishDouble(pub_V_angle, V_ang_init)
    h.helicsPublicationPublishDouble(pub_next_dt, dt_sched)
    h.helicsPublicationPublishBoolean(pub_skip_emt, skip_sched)
    log.info(f"Initial boundary voltage published: "
             f"V={V_mag_init:.5f}pu, angle={V_ang_init:.3f}deg")

    # 4. Enter execution mode -- waits for all federates to be ready
    log.info("Entering HELICS execution mode (waiting for PSCAD federate)...")
    h.helicsFederateEnterExecutingMode(fed)
    log.info(f"Co-simulation started (Adaptive RLS: {ADAPTIVE_STEPPING}, "
             f"range: [{sched_cfg.dt_min_s*1e3:.1f}ms - {sched_cfg.dt_max_s*1e3:.1f}ms])")

    # 5. Time-stepping loop
    current_time = 0.0
    step         = 0
    emt_calls    = 0
    emt_skipped  = 0
    prev_v_mag   = V_mag_init
    prev_v_ang   = V_ang_init
    p_apply, q_apply = DC_P_MW, DC_Q_MVAR

    prev_phi = BoundarySurrogateRLS.build_features(
        v_mag=V_mag_init,
        dv=0.0,
        v_ang_rad=math.radians(V_ang_init),
        dangle_rad=0.0,
        p_prev=p_apply,
        q_prev=q_apply,
        dt=dt_curr,
        ctx=ctx,
    )

    wallclock_start = time.time()
    MAX_WALLCLOCK_S = 120.0   # Hard 2-minute time gate

    while current_time < TOTAL_SIM_TIME - 1e-9:
        if time.time() - wallclock_start > MAX_WALLCLOCK_S:
            log.warning(f"Time gate reached ({MAX_WALLCLOCK_S:.0f}s). Finalizing PSS/E federate cleanly at t={current_time:.3f}s...")
            break

        # Request next timestep from HELICS broker
        target_t = min(TOTAL_SIM_TIME, current_time + dt_curr)
        granted_time = h.helicsFederateRequestTime(fed, target_t)
        step += 1

        if skip_curr:
            emt_skipped += 1
            p_apply, q_apply, unc = surrogate.predict(prev_phi)
            surprise_mw = 0.0
        else:
            emt_calls += 1
            P_meas = h.helicsInputGetDouble(sub_P)
            Q_meas = h.helicsInputGetDouble(sub_Q)
            err_p, _ = surrogate.update(prev_phi, P_meas, Q_meas)
            surprise_mw = err_p
            p_apply, q_apply = P_meas, Q_meas
            _, _, unc = surrogate.predict(prev_phi)

        # ── Update datacenter load in PSS/E ──────────────────────────────
        apply_datacenter_load(p_apply, q_apply)

        # ── Advance PSS/E to granted_time ────────────────────────────────
        ierr_run = step_psse(granted_time)
        if ierr_run != 0:
            log.warning(f"  PSS/E run step warning at t={granted_time:.3f}s: ierr={ierr_run}")

        # ── Read updated boundary voltage ─────────────────────────────────
        V_mag, V_angle = get_boundary_voltage()
        dv = V_mag - prev_v_mag
        dangle = math.radians(V_angle - prev_v_ang)

        # Slack frequency innovation
        df_hz = 0.0
        try:
            ierr_f, f_slack = psspy.busdat(7098, 'PU')
            if ierr_f == 0 and f_slack > 0.5:
                df_hz = (f_slack - 1.0) * 60.0
        except Exception:
            pass

        phi_next = BoundarySurrogateRLS.build_features(
            v_mag=V_mag,
            dv=dv,
            v_ang_rad=math.radians(V_angle),
            dangle_rad=dangle,
            p_prev=p_apply,
            q_prev=q_apply,
            dt=dt_sched,
            ctx=ctx,
        )
        _, _, unc_next = surrogate.predict(phi_next)

        # Pipeline 1-step-ahead action so PSCAD receives schedule in advance
        t_future = granted_time + dt_sched
        dt_next_sched, skip_next_sched, diag = policy.select_action(
            t_now=t_future,
            t_end=TOTAL_SIM_TIME,
            surprise_mw=surprise_mw,
            uncertainty=unc_next,
            dv_pu=dv,
            df_hz=df_hz,
        )

        if step % 10 == 0 or skip_curr != skip_sched:
            log.info(f"t={granted_time:.3f}s | dt={dt_curr*1e3:.1f}ms | "
                     f"skip_emt={skip_curr} | P={p_apply:.2f}MW, Q={q_apply:.2f}MVAR | "
                     f"V={V_mag:.5f}pu | unc={unc:.3f}")

        # Advance schedule pipeline
        dt_curr = dt_sched
        skip_curr = skip_sched
        dt_sched = dt_next_sched
        skip_sched = skip_next_sched

        prev_v_mag = V_mag
        prev_v_ang = V_angle
        prev_phi   = phi_next

        # ── Publish to PSCAD ─────────────────────────────────────────────
        h.helicsPublicationPublishDouble(pub_V_mag,   V_mag)
        h.helicsPublicationPublishDouble(pub_V_angle, V_angle)
        h.helicsPublicationPublishDouble(pub_next_dt, dt_sched)
        h.helicsPublicationPublishBoolean(pub_skip_emt, skip_sched)

        current_time = granted_time

    log.info(f"Co-simulation steps: total={step}, emt_calls={emt_calls}, "
             f"emt_skipped={emt_skipped} (bypass_ratio={emt_skipped/max(1,step)*100:.1f}%)")

    # 6. Close output file and finalize
    try:
        psspy.chsb(0, 0, [-1, -1, -1, 1, 1, 0])
        psspy.change_channel_out_file("")
    except Exception as e:
        log.warning(f"Close channel output warning: {e}")
    log.info(f"PSS/E channel output written to: {OUTPUT_FILE}")

    log.info("Co-simulation complete. Finalizing PSS/E federate...")
    h.helicsFederateFinalize(fed)
    h.helicsFederateFree(fed)
    h.helicsCloseLibrary()
    log.info("PSS/E federate finalized cleanly")

    # 7. Post-process and print key results
    log.info("Reading output channels for summary...")
    try:
        chnfobj = dyntools.CHNF(OUTPUT_FILE)
        _, ch_id, ch_data = chnfobj.get_data()

        time_vec = ch_data['time']
        freq_sl  = ch_data.get(3, [])   # slack bus frequency

        if freq_sl:
            f_min = min(freq_sl)
            f_max = max(freq_sl)
            log.info(f"Slack bus frequency range: "
                     f"{f_min*60:.4f} -- {f_max*60:.4f} Hz")
            log.info(f"Frequency nadir: {f_min*60:.4f} Hz "
                     f"(deviation: {(f_min-1.0)*60:.4f} Hz)")
    except Exception as e:
        log.warning(f"Could not read output file: {e}")

    try:
        psspy.pssehalt_2()
    except Exception:
        pass


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_cosim()