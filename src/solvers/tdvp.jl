import ITensorNetworks as itn
using Printf

@kwdef mutable struct TDVPProblem{State}
  state::State
  operator
  #current_time::Number = 0.0
end

state(tdvp::TDVPProblem) = tdvp.state
operator(tdvp::TDVPProblem) = tdvp.operator

function set(tdvp::TDVPProblem; state=state(tdvp), operator=operator(tdvp))
  return TDVPProblem(; state, operator)
end

function region_plan(tdvp::TDVPProblem; nsites, time_step, sweep_kwargs...)
  return tdvp_regions(state(tdvp), time_step; nsites, sweep_kwargs...)
end

function updater!(tdvp::TDVPProblem, local_tensor, region; outputlevel, kws...)
  local_tensor, info = exponentiate_updater(operator(tdvp), local_tensor; kws...)
  return local_tensor
end

function applyexp(
  H, init_state, time_points; updater_kwargs=(;), inserter_kwargs=(;), outputlevel=0, kws...
)
  init_prob = TDVPProblem(; state=copy(init_state), operator=itn.ProjTTN(H))
  time_steps = diff([0.0, time_points...])
  common_sweep_kwargs = (; outputlevel, updater_kwargs, inserter_kwargs)
  kwargs_array = [(; common_sweep_kwargs..., time_step=t) for t in time_steps]
  sweep_iter = sweep_iterator(init_prob, kwargs_array)
  converged_prob = alternating_update(sweep_iter; outputlevel, kws...)
  return state(converged_prob)
end
