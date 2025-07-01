using Printf: @printf
import ConstructionBase: setproperties

@kwdef mutable struct ApplyExpProblem{State}
  state::State
  operator
  current_time::Number = 0.0
end

ITensorNetworks.state(tdvp::ApplyExpProblem) = tdvp.state
operator(tdvp::ApplyExpProblem) = tdvp.operator
current_time(tdvp::ApplyExpProblem) = tdvp.current_time

function region_plan(tdvp::ApplyExpProblem; nsites, time_step, sweep_kwargs...)
  return tdvp_regions(state(tdvp), time_step; nsites, sweep_kwargs...)
end

function updater(
  T::ApplyExpProblem,
  local_state,
  region_iterator;
  nsites,
  time_step,
  solver=runge_kutta_solver,
  outputlevel,
  kws...,
)
  local_state, info = solver(x->optimal_map(operator(T), x), time_step, local_state; kws...)

  if nsites==1
    curr_reg = current_region(region_iterator)
    next_reg = next_region(region_iterator)
    if !isnothing(next_reg) && next_reg != curr_reg
      next_edge = first(edge_sequence_between_regions(state(T), curr_reg, next_reg))
      v1, v2 = src(next_edge), dst(next_edge)
      psi = copy(state(T))
      psi[v1], R = qr(local_state, uniqueinds(local_state, psi[v2]))
      shifted_operator = position(operator(T), psi, NamedEdge(v1=>v2))
      R_t, _ = solver(x->optimal_map(shifted_operator, x), -time_step, R; kws...)
      local_state = psi[v1]*R_t
    end
  end

  curr_time = current_time(T) + time_step
  T = setproperties(T; current_time=curr_time)

  return T, local_state
end

function applyexp_sweep_printer(
  region_iterator; outputlevel, sweep, nsweeps, process_time=identity, kws...
)
  if outputlevel >= 1
    T = problem(region_iterator)
    @printf("  Current time = %s, ", process_time(current_time(T)))
    @printf("maxlinkdim=%d", maxlinkdim(state(T)))
    println()
    flush(stdout)
  end
end

function applyexp(
  init_prob,
  exponents;
  extracter_kwargs=(;),
  updater_kwargs=(;),
  inserter_kwargs=(;),
  outputlevel=0,
  nsites=1,
  tdvp_order=4,
  sweep_printer=applyexp_sweep_printer,
  kws...,
)
  time_steps = diff([0.0, exponents...])[2:end]
  sweep_kws = (;
    outputlevel, extracter_kwargs, inserter_kwargs, nsites, tdvp_order, updater_kwargs
  )
  kws_array = [(; sweep_kws..., time_step=t) for t in time_steps]
  sweep_iter = sweep_iterator(init_prob, kws_array)
  converged_prob = sweep_solve(sweep_iter; outputlevel, sweep_printer, kws...)
  return state(converged_prob)
end

function applyexp(H, init_state, exponents; kws...)
  init_prob = ApplyExpProblem(;
    state=permute_indices(init_state), operator=ProjTTN(permute_indices(H))
  )
  return applyexp(init_prob, exponents; kws...)
end

process_real_times(z) = round(-imag(z); digits=10)

function tdvp(
  H,
  init_state,
  time_points;
  process_time=process_real_times,
  sweep_printer=(a...; k...)->applyexp_sweep_printer(a...; process_time, k...),
  kws...,
)
  exponents = [-im*t for t in time_points]
  return applyexp(H, init_state, exponents; sweep_printer, kws...)
end
