@kwdef mutable struct FittingProblem{State, OverlapNetwork}
    state::State
    overlapnetwork::OverlapNetwork
    squared_scalar::Number = 0
end

squared_scalar(F::FittingProblem) = F.squared_scalar
state(F::FittingProblem) = F.state
overlapnetwork(F::FittingProblem) = F.overlapnetwork

function set(F::FittingProblem; state = state(F), overlapnetwork = overlapnetwork(F), squared_scalar = squared_scalar(F))
    return FittingProblem(; state, linearformnetwork, squared_scalar)
end

function fit_tensornetwork(tn::AbstractITensorNetwork, init_state::AbstractITensorNetwork, vertex_partitioning)
    overlap_bpc = BeliefPropagationCache(inner_network(tn, init_state), vertex_partitioning)
    init_prob = FittingProblem(; state = copy(init_state), overlapnetwork = overlap_bpc)
    common_sweep_kwargs = (; nsites, outputlevel, updater_kwargs, inserter_kwargs)
    kwargs_array = [(; common_sweep_kwargs..., sweep = s) for s in 1:nsweeps]
    sweep_iter = sweep_iterator(init_prob, kwargs_array)
    converged_prob = alternating_update(sweep_iter; outputlevel, kws...)
    return squared_scalar(converged_prob), state(converged_prob)
end