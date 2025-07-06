using Printf: @printf

@kwdef mutable struct ApplyExpProblem{State}
  operator
  state::State
  current_time::Number = 0.0
end

ITensorNetworks.state(A::ApplyExpProblem) = A.state
operator(A::ApplyExpProblem) = A.operator
current_time(A::ApplyExpProblem) = A.current_time

function set_operator(A::ApplyExpProblem, operator)
  ApplyExpProblem(operator, A.state, A.current_time)
end
set_state(A::ApplyExpProblem, state) = ApplyExpProblem(A.operator, state, A.current_time)
function set_current_time(A::ApplyExpProblem, current_time)
  ApplyExpProblem(A.operator, A.state, current_time)
end

function region_plan(tdvp::ApplyExpProblem; nsites, time_step, sweep_kwargs...)
  return tdvp_regions(state(tdvp), time_step; nsites, sweep_kwargs...)
end

function update(
  prob::ApplyExpProblem,
  local_state,
  region_iterator;
  nsites,
  time_step,
  solver=runge_kutta_solver,
  outputlevel,
  kws...,
)
  local_state, info = solver(
    x->optimal_map(operator(prob), x), time_step, local_state; kws...
  )

  if nsites==1
    curr_reg = current_region(region_iterator)
    next_reg = next_region(region_iterator)
    if !isnothing(next_reg) && next_reg != curr_reg
      next_edge = first(edge_sequence_between_regions(state(prob), curr_reg, next_reg))
      v1, v2 = src(next_edge), dst(next_edge)
      psi = copy(state(prob))
      psi[v1], R = qr(local_state, uniqueinds(local_state, psi[v2]))
      shifted_operator = position(operator(prob), psi, NamedEdge(v1=>v2))
      R_t, _ = solver(x->optimal_map(shifted_operator, x), -time_step, R; kws...)
      local_state = psi[v1]*R_t
    end
  end

  prob = set_current_time(prob, current_time(prob)+time_step)

  return prob, local_state
end

function sweep_printer(
  problem::ApplyExpProblem; outputlevel, sweep, nsweeps, process_time=identity, kws...
)
  if outputlevel >= 1
    @printf("  Current time = %s, ", process_time(current_time(problem)))
    @printf("maxlinkdim=%d", maxlinkdim(state(problem)))
    println()
    flush(stdout)
  end
end

function applyexp(
  operator::AbstractITensorNetwork,
  exponents,
  init_state::AbstractITensorNetwork;
  extract_kwargs=(;),
  update_kwargs=(;),
  insert_kwargs=(;),
  outputlevel=0,
  nsites=1,
  tdvp_order=4,
  kws...,
)
  init_prob = ApplyExpProblem(;
    state=align_indices(init_state), operator=ProjTTN(align_indices(operator))
  )
  time_steps = diff([zero(eltype(exponents)); exponents])[2:end]
  sweep_kws = (;
    outputlevel, extract_kwargs, insert_kwargs, nsites, tdvp_order, update_kwargs
  )
  kws_array = [(; sweep_kws..., time_step=t) for t in time_steps]
  sweep_iter = sweep_iterator(init_prob, kws_array)
  converged_prob = sweep_solve(sweep_iter; outputlevel, kws...)
  return state(converged_prob)
end

process_real_times(z) = round(-imag(z); digits=10)

function time_evolve(
  operator,
  time_points,
  init_state;
  process_time=process_real_times,
  sweep_printer=(a...; k...)->sweep_printer(a...; process_time, k...),
  kws...,
)
  exponents = [-im*t for t in time_points]
  return applyexp(operator, exponents, init_state; sweep_printer, kws...)
end
