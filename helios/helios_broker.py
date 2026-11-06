"""
helios_broker.py
═══════════════════════════════════════════════════════════════════════════════
HELIOS Co-Simulation — Standalone Broker

Use this when you want to start the broker manually and launch federates
separately (useful for debugging one federate at a time).

Workflow:
    Terminal 1:  python helios_broker.py          (start broker first)
    Terminal 2:  python helios_psse_federate.py   (start PSS/E federate)
    Terminal 3:  python helios_pscad_federate.py  (start PSCAD federate)

Alternatively, use the runner for a single-command launch:
    helics run --path=helios_runner.json

The broker manages time synchronization between PSS/E and PSCAD.
It grants time to each federate only when BOTH have requested it,
which is what keeps the two simulations synchronized at each 10ms exchange.
═══════════════════════════════════════════════════════════════════════════════
"""

import helics as h
import logging
import time

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [BROKER] %(message)s"
)
log = logging.getLogger(__name__)


def run_broker():
    """
    Create and run the HELICS broker.
    Waits for exactly 2 federates: psse_federate and pscad_federate.
    Broker stays alive until both federates have finalized.
    """

    # Create broker expecting 2 federates
    # "--federates=2" tells the broker to wait for exactly 2 connections
    # before allowing the co-simulation to proceed
    broker = h.helicsCreateBroker(
        "zmq",              # core type -- must match federates
        "helios_broker",    # broker name -- must be unique
        "--federates=2"     # wait for exactly 2 federates
    )

    log.info("HELIOS broker started")
    log.info("Waiting for 2 federates to connect:")
    log.info("  1. psse_federate  (PSS/E / ACTIVSg2000)")
    log.info("  2. pscad_federate (PSCAD / DOE datacenter model)")
    log.info("Boundary bus: 5385, DALLAS 3 1, 161kV")

    # Keep broker alive while federates are running
    # helicsBrokerIsConnected returns False once all federates have finalized
    while h.helicsBrokerIsConnected(broker):
        time.sleep(0.5)   # poll every 500ms

    # Clean up
    log.info("All federates finalized. Shutting down broker...")
    h.helicsBrokerDisconnect(broker)
    h.helicsBrokerFree(broker)
    h.helicsCloseLibrary()
    log.info("Broker shut down cleanly")


if __name__ == "__main__":
    run_broker()