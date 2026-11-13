"""
helios_pscad_federate.py
═══════════════════════════════════════════════════════════════════════════════
HELIOS Co-Simulation — PSCAD Federate
DOE Datacenter EMT Model (Electromagnetic Transient Side)

Boundary Bus:  5385  DALLAS 3 1  161.0 kV
Pre-disturbance: V = 1.00559 pu,  angle = -72.489 deg

Supports:
  - Legacy fixed-interval mode (EXCHANGE_INTERVAL = 10ms)
  - Adaptive variable-interval + surrogate-bypass mode via HELICS
    subscriptions `psse_federate/next_dt` and `psse_federate/skip_emt`
  - Both `zmq` (multi-process) and `inproc` (single-process runner) cores
═══════════════════════════════════════════════════════════════════════════════
"""

import helics as h
import logging
import math
import os
import socket
import struct
import time
from typing import Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PSCAD] %(message)s",
    handlers=[
        logging.FileHandler("helios_pscad.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

PSCAD_PROJECT_NAME = "HELIOS_datacenter"
PSCAD_CASE_NAME    = "HELIOS_datacenter"
PSCAD_VSOURCE_PATH = "Main:BoundaryVoltageSource"

PSCAD_OUTPUT_P     = "P_PCC"
PSCAD_OUTPUT_Q     = "Q_PCC"

V_BASE_KV          = 161.0
S_BASE_MVA         = 100.0

V_INIT_PU          = 1.00559
V_INIT_DEG         = -72.489

EXCHANGE_INTERVAL  = float(os.environ.get("HELIOS_EXCHANGE_INTERVAL", "10e-3"))
TOTAL_SIM_TIME     = float(os.environ.get("HELIOS_TOTAL_SIM_TIME", "10.0"))
ADAPTIVE_STEPPING  = os.environ.get("HELIOS_ADAPTIVE_STEPPING", "1") == "1"


def stub_step_datacenter(
    stub_state: dict,
    v_mag_pu: float,
    target_time_s: float,
    rated_p_mw: float,
    rated_q_mvar: float,
    profile: str,
    simulate_delay: bool = True,
):
    """Dynamic datacenter/converter load response when mhi.pscad is unavailable."""
    dt = max(1e-4, target_time_s - stub_state["last_t"])
    stub_state["last_t"] = target_time_s

    if profile == "constant_power":
        p_target = rated_p_mw * min(1.0, v_mag_pu / 0.70)
        q_target = rated_q_mvar
        tau = 0.015
    elif profile == "inverter_gfl":
        v_dep = max(0.0, 0.92 - v_mag_pu)
        p_target = rated_p_mw * min(1.05, (v_mag_pu / V_INIT_PU) ** 0.6)
        q_target = rated_q_mvar - 180.0 * v_dep
        tau = 0.020
    else:
        if v_mag_pu < 0.88:
            scale = max(0.25, (v_mag_pu / 0.88) ** 1.4)
            p_target = rated_p_mw * scale
            q_target = rated_q_mvar * scale + 25.0 * (0.88 - v_mag_pu)
        else:
            p_target = rated_p_mw * (1.0 + 0.08 * (v_mag_pu - V_INIT_PU))
            q_target = rated_q_mvar * (v_mag_pu / V_INIT_PU) ** 1.5
        tau = 0.025

    alpha = 1.0 - math.exp(-dt / tau)
    stub_state["p_mw"] += alpha * (p_target - stub_state["p_mw"])
    stub_state["q_mvar"] += alpha * (q_target - stub_state["q_mvar"])
    return stub_state["p_mw"], stub_state["q_mvar"]


# ─── PSCAD Automation Interface ──────────────────────────────────────────────

def connect_pscad():
    """ComFab co-simulation uses direct TCP socket (Client ID 40001).
    Bypasses mhi.pscad COM automation to prevent IPC connection hangs.
    """
    log.info("ComFab mode active: direct high-speed socket on port 40001 (Application_1)")
    return None, None


class PSCADSocketBridge:
    """
    TCP Socket Bridge to PSCAD Application_1 co-simulation component.
    Client ID 40001 uses port 40001.
    Receives [P, Q] (2 doubles = 16 bytes) from PSCAD Power Meter.
    Sends [Phase_deg, V_kV] (2 doubles = 16 bytes) to PSCAD RRL Source.
    """
    def __init__(self, port: int = 40001, host: str = "127.0.0.1"):
        self.port = port
        self.host = host
        self.server_sock = None
        self.client_sock = None
        self.connected = False
        self.is_server = False

    def start(self):
        """Listen for PSCAD Application_1 (Client ID 40001) connection."""
        try:
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_sock.bind((self.host, self.port))
            self.server_sock.listen(1)
            self.server_sock.settimeout(0.05)
            self.is_server = True
            log.info(f"PSCAD Socket Bridge listening on {self.host}:{self.port} (ready for Application_1)")
        except Exception as e:
            # Fallback: if port is already open by PSCAD server, connect as client
            self.is_server = False
            try:
                self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_sock.settimeout(0.1)
                self.client_sock.connect((self.host, self.port))
                self.connected = True
                log.info(f"PSCAD Socket Bridge connected to PSCAD on {self.host}:{self.port}")
            except Exception as ce:
                log.info(f"PSCAD Application_1 not active yet ({ce}) -- will fall back to dynamic model until connected")

    def exchange(self, v_mag_pu: float, v_angle_deg: float) -> Optional[Tuple[float, float]]:
        """Send boundary [Phase_deg, V_kV] and receive measured [P_MW, Q_MVAR]."""
        v_mag_kv = v_mag_pu * V_BASE_KV

        if self.is_server and not self.connected:
            try:
                client, addr = self.server_sock.accept()
                client.settimeout(0.1)
                self.client_sock = client
                self.connected = True
                log.info(f"PSCAD Application_1 connected successfully from {addr}!")
            except (socket.timeout, BlockingIOError):
                return None
            except Exception as e:
                return None

        if not self.connected or not self.client_sock:
            return None

        try:
            # Output 1 = Ph (Angle in deg), Output 2 = V (Voltage in kV)
            out_bytes = struct.pack("<2d", float(v_angle_deg), float(v_mag_kv))
            self.client_sock.sendall(out_bytes)

            in_bytes = self.client_sock.recv(16)
            if len(in_bytes) >= 16:
                P_val, Q_val = struct.unpack("<2d", in_bytes[:16])
                # Normalize if PSCAD power meter outputs in W / VAR instead of MW / MVAR
                if abs(P_val) > 10000:
                    P_val /= 1e6
                if abs(Q_val) > 10000:
                    Q_val /= 1e6
                return P_val, Q_val
        except Exception as e:
            log.debug(f"Socket exchange timeout: {e}")
            return None

        return None

    def close(self):
        if self.client_sock:
            try:
                self.client_sock.close()
            except Exception:
                pass
        if self.server_sock:
            try:
                self.server_sock.close()
            except Exception:
                pass


def pscad_apply_inputs(project, v_mag_pu, v_angle_deg):
    """Apply boundary voltage inputs to PSCAD voltage source component if automation available."""
    if project is None:
        return
    v_mag_kv = v_mag_pu * V_BASE_KV
    try:
        vsource = project.find(PSCAD_VSOURCE_PATH)
        if vsource:
            vsource.parameters(Vrms=v_mag_kv, Angle=v_angle_deg)
    except Exception as e:
        pass


def pscad_run_one_step(
    socket_bridge,
    pscad,
    project,
    target_time_s,
    stub_state,
    v_mag_pu=V_INIT_PU,
    v_angle_deg=V_INIT_DEG,
    rated_p_mw=150.0,
    rated_q_mvar=30.6,
    profile="datacenter_vfd",
    simulate_delay=True,
):
    """Step forward: checks live Application_1 socket first, then fallback stub."""
    if socket_bridge is not None:
        meas = socket_bridge.exchange(v_mag_pu, v_angle_deg)
        if meas is not None:
            return meas

    return stub_step_datacenter(
        stub_state, v_mag_pu, target_time_s,
        rated_p_mw, rated_q_mvar, profile, simulate_delay=simulate_delay
    )


# ─── HELICS Federate Setup & Runner ──────────────────────────────────────────

def create_pscad_federate(
    core_type: str = "zmq",
    broker_name: Optional[str] = None,
    adaptive: bool = ADAPTIVE_STEPPING,
    exchange_interval: float = EXCHANGE_INTERVAL,
    rated_p_mw: float = 150.0,
    rated_q_mvar: float = 30.6,
):
    """Create and register the PSCAD HELICS value federate."""
    fedinfo = h.helicsCreateFederateInfo()
    h.helicsFederateInfoSetCoreName(fedinfo, f"pscad_core_{id(fedinfo)}")
    h.helicsFederateInfoSetCoreTypeFromString(fedinfo, core_type)
    init_str = "--federates=1" + (f" --broker={broker_name}" if broker_name else "")
    h.helicsFederateInfoSetCoreInitString(fedinfo, init_str)

    period = 0.0 if adaptive else exchange_interval
    h.helicsFederateInfoSetTimeProperty(fedinfo, h.HELICS_PROPERTY_TIME_PERIOD, period)
    h.helicsFederateInfoSetFlagOption(fedinfo, h.HELICS_FLAG_SLOW_RESPONDING, True)
    h.helicsFederateInfoSetFlagOption(fedinfo, h.HELICS_FLAG_UNINTERRUPTIBLE, True)

    fed = h.helicsCreateValueFederate("pscad_federate", fedinfo)

    pub_P = h.helicsFederateRegisterGlobalPublication(
        fed, "pscad_federate/P_injection", h.HELICS_DATA_TYPE_DOUBLE, "MW"
    )
    pub_Q = h.helicsFederateRegisterGlobalPublication(
        fed, "pscad_federate/Q_injection", h.HELICS_DATA_TYPE_DOUBLE, "MVAR"
    )

    sub_V_mag = h.helicsFederateRegisterSubscription(
        fed, "psse_federate/boundary_V_mag", "pu"
    )
    sub_V_angle = h.helicsFederateRegisterSubscription(
        fed, "psse_federate/boundary_V_angle", "deg"
    )
    sub_next_dt = h.helicsFederateRegisterSubscription(
        fed, "psse_federate/next_dt", "s"
    )
    h.helicsInputSetDefaultDouble(sub_next_dt, exchange_interval)

    sub_skip_emt = h.helicsFederateRegisterSubscription(
        fed, "psse_federate/skip_emt", ""
    )
    h.helicsInputSetDefaultBoolean(sub_skip_emt, False)

    h.helicsInputSetDefaultDouble(sub_V_mag, V_INIT_PU)
    h.helicsInputSetDefaultDouble(sub_V_angle, V_INIT_DEG)

    return fed, pub_P, pub_Q, sub_V_mag, sub_V_angle, sub_next_dt, sub_skip_emt


def run_cosim(
    total_time: float = TOTAL_SIM_TIME,
    exchange_interval: float = EXCHANGE_INTERVAL,
    adaptive: bool = ADAPTIVE_STEPPING,
    core_type: str = "zmq",
    broker_name: Optional[str] = None,
    rated_p_mw: float = 150.0,
    rated_q_mvar: float = 30.6,
    profile: str = "datacenter_vfd",
    simulate_delay: bool = True,
    close_library: bool = True,
):
    """Main co-simulation loop supporting fixed and adaptive exchange steps."""
    pscad, project = connect_pscad()
    socket_bridge = PSCADSocketBridge(port=40001)
    socket_bridge.start()
    fed, pub_P, pub_Q, sub_V_mag, sub_V_angle, sub_next_dt, sub_skip_emt = create_pscad_federate(
        core_type=core_type,
        broker_name=broker_name,
        adaptive=adaptive,
        exchange_interval=exchange_interval,
        rated_p_mw=rated_p_mw,
        rated_q_mvar=rated_q_mvar,
    )

    stub_state = {"p_mw": rated_p_mw, "q_mvar": rated_q_mvar, "last_t": 0.0}

    h.helicsFederateEnterInitializingMode(fed)
    h.helicsPublicationPublishDouble(pub_P, rated_p_mw)
    h.helicsPublicationPublishDouble(pub_Q, rated_q_mvar)
    h.helicsFederateEnterExecutingMode(fed)

    current_time = 0.0
    dt_curr = exchange_interval
    skip_curr = False
    step = 0
    emt_calls = 0
    emt_skipped = 0
    last_P, last_Q = rated_p_mw, rated_q_mvar
    wallclock_start = time.time()
    MAX_WALLCLOCK_S = 120.0   # Hard 2-minute time gate

    while current_time < total_time - 1e-9:
        if time.time() - wallclock_start > MAX_WALLCLOCK_S:
            log.warning(f"Time gate reached ({MAX_WALLCLOCK_S:.0f}s). Finalizing PSCAD federate cleanly...")
            break
        target_t = min(total_time, current_time + dt_curr)
        granted_time = h.helicsFederateRequestTime(fed, target_t)
        step += 1

        v_mag = h.helicsInputGetDouble(sub_V_mag)
        v_angle = h.helicsInputGetDouble(sub_V_angle)

        if skip_curr:
            emt_skipped += 1
            stub_state["last_t"] = granted_time
            P_MW, Q_MVAR = last_P, last_Q
        else:
            emt_calls += 1
            pscad_apply_inputs(project, v_mag, v_angle)
            P_MW, Q_MVAR = pscad_run_one_step(
                socket_bridge, pscad, project, granted_time, stub_state,
                v_mag_pu=v_mag, v_angle_deg=v_angle, rated_p_mw=rated_p_mw, rated_q_mvar=rated_q_mvar,
                profile=profile, simulate_delay=simulate_delay,
            )
            last_P, last_Q = P_MW, Q_MVAR

        if adaptive:
            nxt = h.helicsInputGetDouble(sub_next_dt)
            dt_curr = nxt if nxt > 1e-5 else exchange_interval
            skip_curr = bool(h.helicsInputGetBoolean(sub_skip_emt))
        else:
            dt_curr = exchange_interval
            skip_curr = False

        if step % 20 == 0:
            log.info(
                f"t={granted_time:.3f}s | V={v_mag:.5f}pu | skip={skip_curr} | "
                f"P={P_MW:.2f}MW, Q={Q_MVAR:.2f}MVAR"
            )

        h.helicsPublicationPublishDouble(pub_P, P_MW)
        h.helicsPublicationPublishDouble(pub_Q, Q_MVAR)
        current_time = granted_time

    socket_bridge.close()
    h.helicsFederateDisconnect(fed)
    h.helicsFederateFree(fed)
    if close_library:
        h.helicsCloseLibrary()
    log.info("PSCAD federate finalized cleanly.")

    return {
        "steps": step,
        "emt_calls": emt_calls,
        "emt_skipped": emt_skipped,
    }


if __name__ == "__main__":
    run_cosim(
        rated_p_mw=float(os.environ.get("HELIOS_RATED_P_MW", "150.0")),
        rated_q_mvar=float(os.environ.get("HELIOS_RATED_Q_MVAR", "30.6")),
        profile=os.environ.get("HELIOS_COMPONENT_PROFILE", "datacenter_vfd"),
    )
