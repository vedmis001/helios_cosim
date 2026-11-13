from numpy import zeros, arange
from models.optimization.Infeasibility import Infeasibility

def compile_sympy(model_dict, ix_dict, opt=False):
    for model in model_dict['models']:
        model.compile_indices(ix_dict)    
    if opt:
        for model in model_dict['models']:
            model.compile_ineq_inds(ix_dict)
            model.compile_cost_inds(ix_dict)
            model.compile_cost_inds(ix_dict, terminal=True)
        for infeas in model_dict['infeasibility_sources']:
            infeas.compile_indices(ix_dict)
            infeas.compile_ineq_inds(ix_dict)
        if 'extra_constraints' in model_dict:
            model_dict['extra_constraints'].compile_indices(ix_dict)
            model_dict['extra_constraints'].ineq_offset = ix_dict['L_ineq'][0]
            # model_dict['extra_constraints'].compile_ineq_inds(ix_dict)

    if 'extra_constraints' in model_dict:
        model_dict['extra_constraints'].compile_initialization()
        model_dict['extra_constraints'].compile_EqualityConstraints()
    for model in model_dict['models']:
        if isinstance(model, list) or isinstance(model, Infeasibility):
            continue
        model.compile_initialization()
        model.compile_EqualityConstraints()
        if opt:
            model.compile_eq_hessian()
            model.compile_init_hessian()
            model.compile_CostFunctions()
            model.compile_CostFunctions(terminal=True)
        # if opt:
        #     model_dict['extra_constraints'].compile_eq_hessian()
        #     model_dict['extra_constraints'].compile_CostFunctions()
        #     model_dict['extra_constraints'].compile_CostFunctions(terminal=True)

def compile_sympy_sensitivity_params(model_dict, ix_dict, study_param_dict):
    col_ctr = 0
    for model_name, param_entry in study_param_dict.items():
        col_ctr = model_dict[model_name].compile_EqJacobianParams(ix_dict, param_entry, col_ctr)

    ix_dict['n_params_under_test'] = col_ctr

    Tvec_param_mask = zeros((ix_dict['n_eq'], col_ctr))
    col_ctr = 0
    for model_name, param_entry in study_param_dict.items():
        model = model_dict[model_name]
        for param_name, _ in param_entry.items():
            if param_name in model.time_constant_pairs:
                dyn_var = getattr(model, model.time_constant_pairs[param_name])
                Tvec_param_mask[dyn_var.i,arange(col_ctr,col_ctr+model.n_dev)] = 1
            col_ctr += model.n_dev
    
    ix_dict['Tvec_param_mask'] = Tvec_param_mask
