"""
helios_pscad_federate.py
═══════════════════════════════════════════════════════════════════════════════
HELIOS Co-Simulation — PSCAD Federate
DOE Datacenter EMT Model (Electromagnetic Transient Side)

Boundary Bus:  5385  DALLAS 3 1  161.0 kV
Pre-disturbance: V = 1.00559 pu,  angle = -72.489 deg
═══════════════════════════════════════════════════════════════════════════════
"""

import helics as h
import logging
import math
import time

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s [PSCAD] %(message)s",
    handlers = [
        logging.FileHandler("helios_pscad.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

# PSCAD project settings
PSCAD_PROJECT_NAME = "HELIOS_datacenter"
PSCAD_CASE_NAME    = "TestCase_VoltageSag_Average1"

# Boundary source component path in PSCAD canvas
# Update to match the exact component name in your PSCAD model
PSCAD_VSOURCE_PATH = "Main:BoundaryVoltageSource"

# Output channel names as defined by E/O blocks in PSCAD
PSCAD_OUTPUT_P    = "P_PCC"     # active power at PCC, MW
PSCAD_OUTPUT_Q    = "Q_PCC"     # reactive power at PCC, MVAR

# Boundary bus electrical parameters
V_BASE_KV         = 161.0       # kV, boundary bus base voltage
S_BASE_MVA        = 100.0       # MVA, system base

# Pre-disturbance initial conditions from PSS/E power flow
V_INIT_PU         = 1.00559     # pu
V_INIT_DEG        = -72.489     # degrees

# HELICS timing
# NOTE: PSCAD's internal EMT timestep is 0.5us
# We do NOT match HELICS exchange period to PSCAD's timestep --
# that would cause millions of HELICS exchanges per second of simulation
# and the communication overhead would dwarf the actual computation.
# Instead we match the HELICS period to PSS/E's transient stability timestep (10ms)
# PSCAD runs freely at 0.5us between each 10ms HELICS exchange
EXCHANGE_INTERVAL = 10e-3       # 10ms -- match PSS/E TS timestep, NOT PSCAD EMT step
TOTAL_SIM_TIME    = 10.0        # seconds -- total co-simulation duration

# ─── PSCAD Automation Interface ──────────────────────────────────────────────

def connect_pscad():
    """
    Connect to a running PSCAD instance.
    PSCAD must be open with the HELIOS project loaded before
    starting the co-simulation.
    """
    log.info("Connecting to PSCAD automation interface...")
    try:
        import mhi.pscad
        pscad = mhi.pscad.connect()

        projects = {p.name: p for p in pscad.projects()}
        if PSCAD_PROJECT_NAME not in projects:
            log.info(f"  Loading project: {PSCAD_PROJECT_NAME}")
            pscad.load(f"{PSCAD_PROJECT_NAME}.pscx")

        project = pscad.project(PSCAD_PROJECT_NAME)
        log.info(f"  Connected to PSCAD project: {PSCAD_PROJECT_NAME}")
        return pscad, project

    except ImportError:
        log.warning("mhi.pscad not installed -- running in stub mode")
        return None, None
    except Exception as e:
        log.error(f"PSCAD connection failed: {e}")
        return None, None


def pscad_apply_inputs(project, v_mag_pu, v_angle_deg):
    """
    Apply boundary voltage inputs to PSCAD.
    Updates the boundary voltage source component each exchange interval.

    v_mag_pu:    voltage magnitude in per unit (received from PSS/E)
    v_angle_deg: voltage angle in degrees (received from PSS/E)

    This is the corrected version of pscad_apply_inputs() from the original
    script -- replaces the current injection / voltage setpoint interface
    with the correct V_mag / V_angle Norton boundary interface.
    """
    if project is None:
        return   # stub mode

    # Convert pu to kV for PSCAD voltage source parameter
    v_mag_kv = v_mag_pu * V_BASE_KV

    try:
        vsource = project.find(PSCAD_VSOURCE_PATH)
        vsource.parameters(
            Vrms  = v_mag_kv,      # line-to-line RMS in kV
            Angle = v_angle_deg,   # phase angle in degrees
        )
    except Exception as e:
        log.warning(f"  Failed to set PSCAD boundary voltage: {e}")


def pscad_run_one_step(pscad, project, target_time_s):
    """
    Step PSCAD forward to target_time_s and read outputs.
    PSCAD runs at its internal 0.5us EMT timestep between calls.

    Returns (P_MW, Q_MVAR) -- what the datacenter is drawing from the grid.
    This is the corrected version of pscad_run_one_step() from the original
    script -- returns P and Q separately (not voltage + power) because
    PSCAD is on the load side of the boundary, not the source side.
    """
    if pscad is None:
        # Stub mode -- return rated values for testing
        time.sleep(EXCHANGE_INTERVAL * 0.1)
        return 150.0, 30.6

    # Advance PSCAD EMT simulation to target time
    # PSCAD runs freely at 0.5us between here and target_time_s
    try:
        pscad.run_to(target_time_s)
    except Exception as e:
        log.warning(f"  PSCAD run_to failed: {e}")
        return 150.0, 30.6

    # Read P and Q from PSCAD output channels (E/O blocks)
    try:
        case   = project.case(PSCAD_CASE_NAME)
        P_MW   = float(case.get_output_value(PSCAD_OUTPUT_P))
        Q_MVAR = float(case.get_output_value(PSCAD_OUTPUT_Q))
        return P_MW, Q_MVAR
    except Exception as e:
        log.warning(f"  PSCAD output read failed: {e} -- using rated values")
        return 150.0, 30.6


# ─── HELICS Federate Setup ───────────────────────────────────────────────────

# we create the federate info object
fedinfo = h.helicsCreateFederateInfo()
h.helicsFederateInfoSetCoreName(fedinfo, "pscad_core")
h.helicsFederateInfoSetCoreTypeFromString(fedinfo, "zmq")

# does NOT start the broker -- broker is started via helios_runner.json
# or helios_broker.py in a separate terminal
h.helicsFederateInfoSetCoreInitString(fedinfo, "--federates=1")

# set time step
# CORRECTION from original: was 50e-6 (matching PSCAD's 0.5us EMT step)
# this is wrong -- it would cause millions of HELICS exchanges per second
# correct value is 10e-3 (10ms) matching PSS/E's transient stability timestep
# PSCAD runs freely at its own 0.5us step between each 10ms HELICS exchange
h.helicsFederateInfoSetTimeProperty(
    fedinfo,
    h.HELICS_PROPERTY_TIME_PERIOD,
    EXCHANGE_INTERVAL   # 10ms, NOT 50e-6
)

# PSCAD EMT runs are slow relative to HELICS wall clock -- prevent timeout
h.helicsFederateInfoSetFlagOption(
    fedinfo,
    h.HELICS_FLAG_SLOW_RESPONDING,
    True
)

# create a value federate (publications and subscriptions)
fed = h.helicsCreateValueFederate("pscad_federate", fedinfo)

# ─── Interface Registration ───────────────────────────────────────────────────

# now, we decide what PSCAD sends and receives
#
# CORRECTION from original:
# Original had PSCAD receiving current injection + voltage setpoint
# and sending terminal voltage + active power.
# This is backwards for EMT/TS co-simulation.
#
# Correct Norton boundary interface:
#   PSCAD RECEIVES from PSS/E: boundary bus V_magnitude (pu) + V_angle (deg)
#   PSCAD SENDS    to   PSS/E: datacenter P_injection (MW) + Q_injection (MVAR)
#
# PSS/E holds the grid voltage (it has the generators and network)
# PSCAD holds the load (it has the datacenter power electronics)
# PSS/E tells PSCAD what voltage the grid is presenting at the boundary
# PSCAD tells PSS/E how much power the datacenter is drawing

# what this federate publishes (PSCAD → PSS/E)
# CORRECTION: was terminal_voltage (complex V) and active_power (W)
# correct: P_injection (MW) and Q_injection (MVAR) -- the load the datacenter draws
pub_P = h.helicsFederateRegisterGlobalPublication(
    fed,
    "pscad_federate/P_injection",   # PSS/E subscribes to this exact name
    h.HELICS_DATA_TYPE_DOUBLE,
    "MW"
)

pub_Q = h.helicsFederateRegisterGlobalPublication(
    fed,
    "pscad_federate/Q_injection",   # PSS/E subscribes to this exact name
    h.HELICS_DATA_TYPE_DOUBLE,
    "MVAR"
)

# what this federate subscribes to (from PSS/E / ACTIVSg2K)
# CORRECTION: was current_injection (A) and voltage_setpoint (V)
# correct: boundary_V_mag (pu) and boundary_V_angle (deg)
# PSS/E publishes the grid-side voltage at Bus 5385 each exchange
sub_V_mag = h.helicsFederateRegisterSubscription(
    fed,
    "psse_federate/boundary_V_mag",     # must match PSS/E pub name exactly
    "pu"
)

sub_V_angle = h.helicsFederateRegisterSubscription(
    fed,
    "psse_federate/boundary_V_angle",   # must match PSS/E pub name exactly
    "deg"
)

# ─── Main Co-Simulation Loop ─────────────────────────────────────────────────

def run_cosim():
    """Main co-simulation loop."""

    # Connect to PSCAD
    pscad, project = connect_pscad()

    # enter initialization mode
    h.helicsFederateEnterInitializingMode(fed)

    # set initial values here before execution starts
    # initial condition: rated steady-state datacenter load
    # these match the 150MW load already in the PSS/E case at Bus 5385
    h.helicsPublicationPublishDouble(pub_P, 150.0)   # 150 MW rated
    h.helicsPublicationPublishDouble(pub_Q,  30.6)   # 30.6 MVAR rated (PF=0.98)

    h.helicsFederateEnterExecutingMode(fed)
    log.info("Co-simulation started")

    # time loop
    # advance HELICS time at PSS/E's transient stability timestep (10ms)
    # since PSCAD timestep is in the us domain, we would have millions of
    # HELICS exchanges per second of simulation if we matched PSCAD's step --
    # this would mean the communication overhead would dwarf the actual computation
    # instead, PSCAD runs freely at 0.5us between each 10ms HELICS exchange

    # total co-sim duration in seconds
    total_time   = TOTAL_SIM_TIME

    # match PSS/E's transient stability timestep
    # CORRECTION from original: was 50e-1 = 5 seconds (too coarse)
    # and also was described as matching PSCAD step (wrong concept)
    # correct value: 10e-3 = 10ms matching PSS/E TS timestep
    time_step    = EXCHANGE_INTERVAL   # 10ms

    current_time = 0.0
    step         = 0

    while current_time < total_time:

        # request next time from broker
        # broker grants time only when BOTH federates have requested it
        # this is what keeps PSS/E and PSCAD synchronized
        granted_time = h.helicsFederateRequestTime(
            fed, current_time + time_step
        )

        step += 1

        # read inputs from ACTIVSg2K (PSS/E)
        # CORRECTION from original: was i_inject (complex A) and v_set (V scalar)
        # correct: v_mag (pu) and v_angle (deg) from PSS/E boundary bus solution
        v_mag   = h.helicsInputGetDouble(sub_V_mag)
        v_angle = h.helicsInputGetDouble(sub_V_angle)

        if step % 10 == 0:   # log every 100ms
            log.info(f"t={granted_time:.3f}s | "
                     f"PSS/E→PSCAD: V={v_mag:.5f}pu, δ={v_angle:.3f}deg")

        # apply to PSCAD boundary voltage source
        # PSCAD coupling code -- updates Main:BoundaryVoltageSource parameters
        pscad_apply_inputs(project, v_mag, v_angle)

        # step PSCAD forward to granted_time
        # PSCAD runs at 0.5us EMT step internally between here and granted_time
        # returns P and Q that the datacenter is drawing from the grid
        P_MW, Q_MVAR = pscad_run_one_step(pscad, project, granted_time)

        if step % 10 == 0:
            log.info(f"t={granted_time:.3f}s | "
                     f"PSCAD→PSS/E: P={P_MW:.3f}MW, Q={Q_MVAR:.3f}MVAR")

        # publish PSCAD outputs to HELICS
        # CORRECTION from original: was complex voltage + scalar power
        # correct: P (MW) and Q (MVAR) separately -- PSS/E needs both
        # to update the datacenter load model at Bus 5385
        h.helicsPublicationPublishDouble(pub_P, P_MW)
        h.helicsPublicationPublishDouble(pub_Q, Q_MVAR)

        current_time = granted_time

    # finalize
    log.info("Co-simulation complete. Finalizing...")
    h.helicsFederateFinalize(fed)
    h.helicsFederateFree(fed)
    h.helicsCloseLibrary()
    log.info("PSCAD federate finalized cleanly")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_cosim()