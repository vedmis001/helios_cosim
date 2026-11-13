import logging
import numpy as np
from numpy import (
    concatenate, arange, zeros, zeros_like, amax, eye, argmax,
    unravel_index, isclose, argwhere, array, amin, pi, ndarray, maximum
)
from time import time
from scipy.optimize import fsolve
from scipy.sparse.linalg import spsolve, splu
from scripts.stamps_and_residues.transient import (
    build_Yts, eval_alg, eval_dyn, eval_ts,
    eval_jac_alg, eval_jac_alg_feas_only, generate_Ypf_ts,
)
from scripts.debugging.jacobian_divided_diffs import generic_divided_diffs
from scripts.debugging.nonlinear_solve import solve_with_scipy
from settings import ts_settings
from scripts.limiting.diode_limiting import constant_diode_limit
from scripts.debugging.index_lookup import *
from scripts.timestep_control import get_next_time, calc_LTE_simple_trap, compute_adaptive_dt
from scripts.event_state_manager import initialize_event_state, set_topology_at_time, get_event_times

logger = logging.getLogger(__name__)


def solve_ts_single_step(model_dict, ix_dict, Vprev, t, dt, Ypf, feas_vals=None):
    """Solve a single timestep of the transient stability DAE system using Newton-Raphson.

    Uses trapezoidal integration for the dynamic (ODE) equations and direct
    algebraic evaluation for the algebraic constraints.  If
    ``ts_settings.use_fsolve_fallback`` is enabled, falls back to
    ``scipy.optimize.fsolve`` when Newton iterations near the limit without
    converging.
    """
    max_iter = 30
    tol = 1e-6
    Tvec = model_dict['Tvec']
    f_prev = eval_dyn(Vprev, (t-dt), model_dict, ix_dict['n_dyn'])
    Fhist = zeros_like(Vprev)
    F = zeros_like(Vprev)
    Fhist[ix_dict['n_alg']:] = f_prev + 2*Tvec/dt*Vprev[ix_dict['n_alg']:]
    iterations = 0
    V = Vprev.copy()
    if feas_vals is not None:
        Yu = eval_jac_alg_feas_only(V, t, model_dict, ix_dict)
    success = False
    while iterations < max_iter and amax(abs(V)) < 1e10:
        iterations += 1
        g = eval_alg(V, t, model_dict, ix_dict['n_alg'])
        f = eval_dyn(V, t, model_dict, ix_dict['n_dyn'])
        F[:ix_dict['n_alg']] = g
        F[ix_dict['n_alg']:] = f - 2*Tvec/dt*V[ix_dict['n_alg']:]
        F += Fhist
        if feas_vals is not None:
            F += Yu@feas_vals
        Y = build_Yts(V, t, dt, Ypf, model_dict, ix_dict)
        if iterations == (max_iter - 1) and getattr(ts_settings, 'use_fsolve_fallback', False):
            V_scipy = fsolve(eval_ts, Vprev.copy(), args=(Vprev, t, dt, model_dict, ix_dict))
            g_scipy = eval_alg(V_scipy, t, model_dict, ix_dict['n_alg'])
            if any(V_scipy[ix_dict['limit_slack_upper']] < 0) or any(V_scipy[ix_dict['limit_slack_lower']] < 0):
                raise ValueError("Limit slack variable has gone negative. Fix diode limiting function.")
            if amax(abs(g_scipy)) < tol:
                V = V_scipy
                success = True
                break
        J = -F + (Y@V)
        Vsol = spsolve(Y, J)
        constant_diode_limit(Vsol, V, model_dict, 1e-6, include_ineq=False)
        if any(Vsol[ix_dict['limit_slack_upper']] < 0) or any(Vsol[ix_dict['limit_slack_lower']] < 0):
            raise ValueError("Limit slack variable has gone negative. Fix diode limiting function.")
        if any(Vsol[model_dict['bus'].Vt.i] < 0):
            Vsol[model_dict['bus'].Vt.i] = abs(Vsol[model_dict['bus'].Vt.i])
        dV = Vsol - V
        err_max = max(amax(abs(dV)), amax(abs(F)))
        if err_max < tol:
            success = True
            break
        dV[model_dict['bus'].theta.i] = np.maximum(-.3, np.minimum(.3, dV[model_dict['bus'].theta.i]))
        V = V + dV
    if amax(abs(V[model_dict['rotor_angle_diffs'].delta_spread.i])) > pi:
        logger.error(f"Synchronism lost at time {t:.2e}. Stopping simulation.")
        success = False
    return V, success


def solve_ts(model_dict, ix_dict, V0, tf, dt_init, t0=0.0, feas_vals=None):
    """Run a full transient stability simulation from t0 to tf."""
    tol = 1e-7
    max_iter = 60
    min_dt = getattr(ts_settings, 'dt_min', 1e-6)

    F = zeros_like(V0)
    Fhist = zeros_like(V0)
    iter_log = [0]
    
    Ypf = generate_Ypf_ts(V0[:ix_dict['n_alg']], model_dict, ix_dict['n_eq'])
    f_prev = eval_dyn(V0, t0, model_dict, ix_dict['n_dyn'])

    if 'event_queue' not in model_dict:
        initialize_event_state(model_dict, t0=t0)
    else:
        set_topology_at_time(model_dict, t0)

    times = [t0]
    waveforms = [V0]
    t = t0+dt_init
    dt_normal = dt_init
    dt_event = min(1e-4, dt_normal)
    t_event_window = 3e-4
    event_state = 0
    termination = 'finished'
    clock_start = time()

    if feas_vals is not None:
        Yu = eval_jac_alg_feas_only(V0, t, model_dict, ix_dict)
    Tvec = model_dict['Tvec']
    event_times = get_event_times(model_dict, t0, tf)
    while times[-1] <= tf and not isclose(times[-1], tf):
        Vprev = waveforms[-1]
        t, next_event_state = get_next_time(event_state, times[-1], dt_normal, dt_event, t_event_window, event_times)
        dt = t - times[-1]
        if dt < min_dt:
            logger.error(f"Unable to converge at t={times[-1]:.4f} even after timestep reduced to minimum level. "
                         "Classifying as voltage collapse and exiting.")
            termination = 'dt_floor'
            break
        Fhist[ix_dict['n_alg']:] = f_prev + 2*Tvec/dt*Vprev[ix_dict['n_alg']:]

        V = Vprev.copy()
        err_max = 10
        iterations = 0
        events_applied, _ = set_topology_at_time(model_dict, t)
        if len(events_applied) > 0:
            for event in events_applied:
                if hasattr(event, 'device_name'):
                    logger.info(f"Event {event.name} applied to {event.device_name}, {event.device_key} at time {t:.3f} s")
                elif hasattr(event, 'g_fault'):
                    logger.info(f"Fault {event.name} applied to bus {event.bus} at time {t:.3f} s")
        success = False
        Ypf = generate_Ypf_ts(V[:ix_dict['n_alg']], model_dict, ix_dict['n_eq'])
        while iterations < max_iter and amax(abs(V)) < 1e10:
            iterations += 1
            if iterations == (max_iter - 1) and getattr(ts_settings, 'use_fsolve_fallback', False):
                V_scipy = fsolve(eval_ts, Vprev.copy(), args=(Vprev, t, dt, model_dict, ix_dict))
                V = V_scipy
            g = eval_alg(V, t, model_dict, ix_dict['n_alg'])
            f = eval_dyn(V, t, model_dict, ix_dict['n_dyn'])
            F[:ix_dict['n_alg']] = g
            F[ix_dict['n_alg']:] = f - 2*Tvec/dt*V[ix_dict['n_alg']:]
            if feas_vals is not None:
                if isinstance(feas_vals, ndarray):
                    F += Yu@feas_vals
                else: 
                    F += Yu@(feas_vals(t))
            F += Fhist
            Y = build_Yts(V, t, dt, Ypf, model_dict, ix_dict)
            J = -F + (Y@V)
            Vsol = spsolve(Y, J)
            constant_diode_limit(Vsol, V, model_dict, 1e-6, include_ineq=False)
            if any(Vsol[ix_dict['limit_slack_upper']] < 0) or any(Vsol[ix_dict['limit_slack_lower']] < 0):
                raise ValueError("Limit slack variable has gone negative. Fix diode limiting function.")
            if any(Vsol[model_dict['bus'].Vt.i] < 0):
                Vsol[model_dict['bus'].Vt.i] = abs(Vsol[model_dict['bus'].Vt.i])
            dV = (Vsol - V)
            iter_err = amax(abs(dV)/maximum(abs(V),.1))
            res_err = amax(abs(F))
            err_max = max(iter_err, res_err)
            if err_max < tol:
                success = True
                break
            dV[model_dict['bus'].theta.i] = np.maximum(-.3, np.minimum(.3, dV[model_dict['bus'].theta.i]))
            V = V + dV
        
        lte_reject = False
        if success and getattr(ts_settings, 'adaptive_dt', False) and len(times) >= 3:
            lte_array = calc_LTE_simple_trap(Vsol, waveforms, t, times)
            dt_new, max_lte = compute_adaptive_dt(dt, lte_array)
            
            if max_lte > getattr(ts_settings, 'tol_lte', 1e-4):
                if dt <= getattr(ts_settings, 'dt_min', 1e-6):
                    logger.warning(f"LTE violated ({max_lte:.2e}) but dt is at lower bound. Forcing accept to prevent infinite failure loop.")
                    lte_reject = False
                else:
                    lte_reject = True
                    success = False
                    logger.warning(f"REJECTED step at t={times[-1]:.4f} | LTE: {max_lte:.2e} > tol | dt scaled: {dt:.2e} -> {dt_new:.2e}")
                    dt_normal = dt_new 
                    dt_event = dt_new
            else:
                if abs(dt_normal - dt_new) > 1e-6 and t > 0.0:
                    logger.info(f"ACCEPTED step at t={t:.4f} | LTE limits ok. dt scaled: {dt_normal:.2e} -> {dt_new:.2e}")
                dt_normal = dt_new

        if success:
            event_state = next_event_state
            if 'theta' in model_dict['bus'].alg_vars:
                theta = V[model_dict['bus'].theta.i]
                theta_wrap = (theta + pi) % (2 * pi) - pi
                Vsol[model_dict['bus'].theta.i] = theta_wrap
            for model in model_dict['controllers']:
                model.update_timer_blocks(Vsol, t)

            iter_log.append(iterations)
            waveforms.append(Vsol)
            times.append(t)
            f_prev = f.copy()
            if ts_settings.track_all_rotor_angle_diffs:
                if amax(abs(V[model_dict['rotor_angle_diffs'].delta_spread.i])) > pi:
                    logger.error(f"Synchronism lost at time {t:.2e}. Stopping simulation.")
                    termination = 'synchronism_lost'
                    break
        else:
            _, events_deactivated = set_topology_at_time(model_dict, times[-1])
            if not lte_reject:
                dt_normal *= 0.5
                dt_event *= 0.5
                logger.warning(f"Step failed to converge. dt reduced to {dt_normal:.3e}")

    clock_finish = time()
    t_elapsed = clock_finish - clock_start
    logger.info("Transient stability simulation completed in %.3e seconds" % t_elapsed)
    waveforms = np.array(waveforms)
    times = np.array(times)
    bus = model_dict['bus']
    Vmag_final = np.hypot(waveforms[-1, bus.Vr.i], waveforms[-1, bus.Vi.i])
    info = {
        'completed': termination == 'finished',
        'collapsed': termination == 'dt_floor',
        'termination': termination,
        't_end': float(times[-1]),
        'tf': float(tf),
        'min_vt_final': float(np.min(Vmag_final)),
        'wall_time': t_elapsed,
    }
    return waveforms, times, info
