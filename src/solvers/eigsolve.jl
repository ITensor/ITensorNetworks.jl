using Printf: @printf
using ITensors: truncerror

@kwdef mutable struct EigsolveProblem{State,Operator} <: AbstractProblem
  operator::Operator
  state::State
  eigenvalue::Number = Inf
  max_truncerror::Real = 0.0
end

eigenvalue(E::EigsolveProblem) = E.eigenvalue
state(E::EigsolveProblem) = E.state
operator(E::EigsolveProblem) = E.operator
max_truncerror(E::EigsolveProblem) = E.max_truncerror

function set_truncation_info!(E::EigsolveProblem; spectrum=nothing)
  if !isnothing(spectrum)
    E.max_truncerror = max(max_truncerror(E), truncerror(spectrum))
  end
  return E
end

function update!(
  local_state,
  region_iterator::RegionIterator{<:EigsolveProblem};
  outputlevel,
  solver=eigsolve_solver,
  kws...,
)
  prob = problem(region_iterator)

  eigval, local_state = solver(ψ -> optimal_map(operator(prob), ψ), local_state; kws...)
  prob.eigenvalue = eigval

  if outputlevel >= 2
    @printf(
      "  Region %s: energy = %.12f\n", current_region(region_iterator), eigenvalue(prob)
    )
  end
  return local_state
end

function default_sweep_callback(
  sweep_iterator::SweepIterator{<:EigsolveProblem}; outputlevel
)
  if outputlevel >= 1
    nsweeps = length(sweep_iterator)
    current_sweep = sweep_iterator.which_sweep
    if length(sweep_iterator) >= 10
      @printf("After sweep %02d/%d ", current_sweep, nsweeps)
    else
      @printf("After sweep %d/%d ", current_sweep, nsweeps)
    end
    @printf("eigenvalue=%.12f", eigenvalue(problem))
    @printf(" maxlinkdim=%d", maxlinkdim(state(problem)))
    @printf(" max truncerror=%d", max_truncerror(problem))
    println()
    flush(stdout)
  end
end

function eigsolve(
  operator,
  init_state;
  nsweeps,
  nsites=1,
  outputlevel=0,
  extract_kwargs=(;),
  update_kwargs=(;),
  insert_kwargs=(;),
  kws...,
)
  init_prob = EigsolveProblem(;
    state=align_indices(init_state), operator=ProjTTN(align_indices(operator))
  )
  sweep_iter = SweepIterator(
    init_prob, nsweeps; nsites, outputlevel, extract_kwargs, update_kwargs, insert_kwargs
  )
  prob = sweep_solve(sweep_iter; outputlevel, kws...)
  return eigenvalue(prob), state(prob)
end

dmrg(operator, init_state; kwargs...) = eigsolve(operator, init_state; kwargs...)
