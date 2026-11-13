from numpy import isclose
import numpy as np
from settings import ts_settings


def get_next_time(event_state, t_prev, dt_normal, dt_event, t_event_window, event_times):
    # event states:
    # 0: not near an event, just take a normal time step
    # 1: an event is about to happen, so make the next t equal to (t event) - (small step)
    # 2: event just occured, so take small time steps for t_event_window
    remaining_event_times = event_times[event_times > t_prev]
    if len(remaining_event_times) == 0:
        next_event_t = None
    else:
        next_event_t = remaining_event_times[0]
    if next_event_t is None:
        t = t_prev + dt_normal
    elif event_state == 0:
        # Cruising state. Step by dt_normal, but cap strictly at next_event_t.
        t = t_prev + dt_normal
        if t >= next_event_t - 1e-12:
            t = next_event_t
            event_state = 1
    elif event_state == 1:
        # Just hit the event exactly. The next step is a post-event step.
        t = t_prev + dt_event
        event_state = 2
    elif event_state == 2:
        # Post-event window stabilization
        time_since_last_event = (t_prev - event_times[event_times <= t_prev])[-1]
        if time_since_last_event > t_event_window or isclose(time_since_last_event, t_event_window):
            event_state = 0
            t = t_prev + dt_normal
            if next_event_t is not None and t >= next_event_t - 1e-12:
                t = next_event_t
                event_state = 1
        else:
            t = t_prev + dt_event
    return t, event_state


def calc_LTE_simple_trap(V_sol, V_hist, t_sol, t_hist):
    # approximate LTE for trapezoidal method using third order divided differences
    # see Pileggi p112 for more info
    if len(t_hist) < 3:
        return np.zeros_like(V_sol)
        
    dt = t_sol - t_hist[-1]
    dtn1 = t_hist[-1] - t_hist[-2]
    dtn2 = t_hist[-2] - t_hist[-3]
    # mu_dt = (dt + dtn1 + dtn2)/3
    DD = (V_sol - V_hist[-1])/dt
    DDn1 = (V_hist[-1] - V_hist[-2])/dtn1
    DDn2 = (V_hist[-2] - V_hist[-3])/dtn2
    third_order_diff = ((DD - DDn1)/(dt + dtn1) - (DDn1 - DDn2)/(dtn1 + dtn2))/(dt + dtn1 + dtn2)
    LTE = -(dt**3)*(third_order_diff/12)
    return LTE


def compute_adaptive_dt(dt_current, LTE_array):
    tol_lte = ts_settings.tol_lte
    dt_max = ts_settings.dt_max
    dt_min = ts_settings.dt_min
    
    max_lte = np.amax(np.abs(LTE_array))
    
    if max_lte < 1e-15:
        max_lte = 1e-15
        
    beta = 0.8 # Safety factor
    scale = beta * (tol_lte / max_lte) ** (1.0/3.0)
    
    # Cap step scaling logically
    scale = np.clip(scale, 0.2, 2.0)
    
    dt_new = dt_current * scale
    dt_new = np.clip(dt_new, dt_min, dt_max)
    
    return dt_new, max_lte
