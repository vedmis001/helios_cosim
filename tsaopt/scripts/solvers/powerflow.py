import logging
from numpy import copy, amax, array, add, zeros_like, load, argmax
from scipy.sparse.linalg import spsolve
from scripts.stamps_and_residues.powerflow import calc_ntwk_residues, eval_Ylin_powerflow, generate_nonlin_mats_ntwk
from scripts.debugging.jacobian_divided_diffs import powerflow_jacobian_divided_diffs
from scripts.debugging.index_lookup import search_ind
from scripts.limiting.filter_limiting import powerflow_filtering

logger = logging.getLogger(__name__)


def solve_pf_tx_stepping(V0, model_dict, use_diode_limiting,
                         h_factor_decrease=0.2):
    """Tx-stepping homotopy."""
    h_factor = 1.0
    max_iter = 100
    iterations = 0
    last_successful_h_factor = 1
    tx_stepping_success = False
    while iterations < max_iter:
        iterations += 1
        Vsol, inner_success = solve_pf(V0, model_dict, use_diode_limiting, h_factor)
        if inner_success:
            if h_factor == 0:
                tx_stepping_success = True
                break
            last_successful_h_factor = h_factor
            h_factor = max(h_factor - h_factor_decrease, 0.0)
            V0 = Vsol
        else:
            if h_factor_decrease < 0.01:
                break
            h_factor_decrease /= 2
            h_factor = last_successful_h_factor - h_factor_decrease
    return Vsol, tx_stepping_success


def network_stepping(V, model_dict, use_diode_limiting):
    ns_res = calc_ntwk_residues(V, model_dict)

    h_factor = 1.0
    h_factor_decrease = 0.2
    max_iter = 100
    iterations = 0
    last_successful_h_factor = 1
    ns_success = False
    while iterations < max_iter:
        iterations += 1
        V_ns = h_factor*ns_res
        Vsol, success = solve_pf(V, model_dict, use_diode_limiting, V_ns)
        if success:
            if h_factor == 0:
                ns_success = True
                break
            last_successful_h_factor = h_factor
            h_factor = max(h_factor-h_factor_decrease, 0.0)
            V = Vsol
        else:
            h_factor = (h_factor + last_successful_h_factor)/2
    return Vsol, ns_success


def solve_pf(V0, model_dict, use_diode_limiting, h_factor=0.0, V_ns=None):
    V = copy(V0)
    Ylin, Jlin = eval_Ylin_powerflow(model_dict, V.shape[0], -1, h_factor)
    err_max = 10
    max_iter = 100
    tol = 1e-6
    iterations = 0
    success = False
    while iterations < max_iter:
        Ynlin, Jnlin = generate_nonlin_mats_ntwk(V, model_dict, False, h_factor)
        Y = Ylin + Ynlin
        J = Jlin + Jnlin
        if V_ns is not None:
            J += V_ns
        V_sol = spsolve(Y, J)
        iterations += 1
        err_max = amax(abs(V_sol - V))
        if err_max < tol:
            V = V_sol.copy()
            success = True
            break
        if use_diode_limiting:
            powerflow_filtering(V_sol, V, model_dict)
        V = V_sol.copy()
    if success:
        res = calc_ntwk_residues(V, model_dict, h_factor)
        if V_ns is not None:
            res += -V_ns
        max_res = amax(abs(res))
        if max_res > 1e-6:
            logger.warning("Warning: powerflow solution maximum residue exceeded threshold: %.3e" % max_res)
    return V, success
