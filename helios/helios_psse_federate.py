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
import psspy
import dyntools
import os
import logging
import math

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
EXCHANGE_INTERVAL = 10e-3   # 10ms -- PSS/E TS timestep
TOTAL_SIM_TIME    = 10.0    # seconds -- total co-simulation duration

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
    """Configure PSS/E output channels for monitoring."""

    psspy.delete_all_plot_channels()

    # Channel 1: Boundary bus voltage magnitude
    psspy.voltage_channel([-1,-1,-1,1], BOUNDARY_BUS, "V_BOUNDARY")

    # Channel 2: Boundary bus frequency
    psspy.freq_channel([-1,-1,-1,1], BOUNDARY_BUS, "FREQ_BOUNDARY")

    # Channel 3: System frequency (slack bus)
    # Bus 7098 is the slack bus from initial power flow
    psspy.freq_channel([-1,-1,-1,1], 7098, "FREQ_SLACK")

    # Channel 4: Datacenter active power
    psspy.load_p_channel([-1,-1,-1,1], BOUNDARY_BUS, DC_LOAD_ID, "P_DC")

    # Channel 5: Datacenter reactive power
    psspy.load_q_channel([-1,-1,-1,1], BOUNDARY_BUS, DC_LOAD_ID, "Q_DC")

    # Open output file
    psspy.chsb(0, 1, [-1,-1,-1,1,1,0], OUTPUT_FILE)
    log.info(f"Output channels configured, writing to: {OUTPUT_FILE}")


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
    # Broker is started separately via: helics run --path=helios_broker.json
    h.helicsFederateInfoSetCoreInitString(fedinfo, "--federates=1")

    # Exchange period = PSS/E TS timestep
    h.helicsFederateInfoSetTimeProperty(
        fedinfo,
        h.HELICS_PROPERTY_TIME_PERIOD,
        EXCHANGE_INTERVAL
    )

    # PSS/E dynamics can be slow -- prevent broker timeout
    h.helicsFederateInfoSetFlagOption(
        fedinfo,
        h.HELICS_FLAG_SLOW_RESPONDING,
        True
    )

    # Time delta: minimum time between granted times
    h.helicsFederateInfoSetTimeProperty(
        fedinfo,
        h.HELICS_PROPERTY_TIME_DELTA,
        EXCHANGE_INTERVAL
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

    # ── Subscriptions (PSCAD → PSS/E) ────────────────────────────────────
    sub_P = h.helicsFederateRegisterSubscription(
        fed,
        "pscad_federate/P_injection",   # must match PSCAD pub name exactly
        "MW"
    )

    sub_Q = h.helicsFederateRegisterSubscription(
        fed,
        "pscad_federate/Q_injection",   # must match PSCAD pub name exactly
        "MVAR"
    )

    log.info("HELICS interfaces registered:")
    log.info("  Publishing:   boundary_V_mag [pu], boundary_V_angle [deg]")
    log.info("  Subscribing:  P_injection [MW], Q_injection [MVAR]")

    return pub_V_mag, pub_V_angle, sub_P, sub_Q


# ─── Main Co-Simulation Loop ─────────────────────────────────────────────────

def run_cosim():
    """Main entry point for the PSS/E HELICS federate."""

    # 1. Initialize PSS/E
    initialize_psse()
    setup_output_channels()

    # 2. Create HELICS federate and register interfaces
    fed = create_psse_federate()
    pub_V_mag, pub_V_angle, sub_P, sub_Q = register_psse_interfaces(fed)

    # 3. Enter initialization mode
    log.info("Entering HELICS initialization mode...")
    h.helicsFederateEnterInitializingMode(fed)

    # Publish initial boundary voltage before execution starts
    # These are the pre-disturbance values from the power flow snapshot
    V_mag_init, V_ang_init = get_boundary_voltage()
    h.helicsPublicationPublishDouble(pub_V_mag,   V_mag_init)
    h.helicsPublicationPublishDouble(pub_V_angle, V_ang_init)
    log.info(f"Initial boundary voltage published: "
             f"V={V_mag_init:.5f}pu, δ={V_ang_init:.3f}deg")

    # 4. Enter execution mode -- waits for all federates to be ready
    log.info("Entering HELICS execution mode (waiting for PSCAD federate)...")
    h.helicsFederateEnterExecutingMode(fed)
    log.info("Co-simulation started")

    # 5. Time-stepping loop
    current_time = 0.0
    step         = 0

    while current_time < TOTAL_SIM_TIME:

        # Request next timestep from HELICS broker
        # Broker grants time only when BOTH federates have requested it
        granted_time = h.helicsFederateRequestTime(
            fed, current_time + EXCHANGE_INTERVAL
        )

        step += 1

        # ── Read P and Q from PSCAD ───────────────────────────────────────
        P_MW   = h.helicsInputGetDouble(sub_P)
        Q_MVAR = h.helicsInputGetDouble(sub_Q)

        if step % 10 == 0:   # log every 100ms to avoid spam
            log.info(f"t={granted_time:.3f}s | "
                     f"PSCAD→PSS/E: P={P_MW:.2f}MW, Q={Q_MVAR:.2f}MVAR")

        # ── Update datacenter load in PSS/E ──────────────────────────────
        apply_datacenter_load(P_MW, Q_MVAR)

        # ── Advance PSS/E one timestep ───────────────────────────────────
        ierr_run = step_psse(granted_time)
        if ierr_run != 0:
            log.warning(f"  PSS/E run step warning at t={granted_time:.3f}s: "
                        f"ierr={ierr_run}")

        # ── Read updated boundary voltage ─────────────────────────────────
        V_mag, V_angle = get_boundary_voltage()

        if step % 10 == 0:
            log.info(f"t={granted_time:.3f}s | "
                     f"PSS/E→PSCAD: V={V_mag:.5f}pu, δ={V_angle:.3f}deg")

        # ── Publish to PSCAD ─────────────────────────────────────────────
        h.helicsPublicationPublishDouble(pub_V_mag,   V_mag)
        h.helicsPublicationPublishDouble(pub_V_angle, V_angle)

        current_time = granted_time

    # 6. Close output file and finalize
    psspy.chsb(0, 0, [-1,-1,-1,1,1,0])
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


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_cosim()