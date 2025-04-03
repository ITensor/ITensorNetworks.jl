@kwdef mutable struct OrthogonalLinearFormProblem{State, LinearFormNetwork}
    state::State
    linearformnetwork::LinearFormNetwork
    squared_scalar::Number = 0
end

squared_scalar(O::OrthogonalLinearFormProblem) = O.squared_scalar
state(O::OrthogonalLinearFormProblem) = O.state
linearformnetwork(O::OrthogonalLinearFormProblem) = O.linearformnetwork

function set(O::OrthogonalLinearFormProblem; state = state(O), linearformnetwork = linearformnetwork(O), squared_scalar = squared_scalar(O))
    return OrthogonalLinearFormProblem(; state, linearformnetwork, squared_scalar)
end

function updater!(O::OrthogonalLinearFormProblem, local_tensor, region; outputlevel, kws...)
    O.squared_scalar, local_tensor = linearform_updater

function maximize_linearformnetwork_sq(linearformnetwork, init_state; nsweeps, nsites=1, outputlevel = 0, update_kwargs = (;), inserter_kwargs = (;), kws...)
    init_prob = OrthogonalLinearFormProblem(; state = copy(init_state), linearformnetwork = linearformnetwork)
    common_sweep_kwargs = (; nsites, outputlevel, updater_kwargs, inserter_kwargs)
    kwargs_array = [(; common_sweep_kwargs..., sweep = s) for s in 1:nsweeps]
    sweep_iter = sweep_iterator(init_prob, kwargs_array)
    converged_prob = alternating_update(sweep_iter; outputlevel, kws...)
    return squared_scalar(converged_prob), state(converged_prob)
end