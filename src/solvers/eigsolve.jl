using Accessors: @set
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

set_operator(E::EigsolveProblem, operator) = (@set E.operator = operator)
set_eigenvalue(E::EigsolveProblem, eigenvalue) = (@set E.eigenvalue = eigenvalue)
set_state(E::EigsolveProblem, state) = (@set E.state = state)
set_max_truncerror(E::EigsolveProblem, truncerror) = (@set E.max_truncerror = truncerror)

function set_truncation_info(E::EigsolveProblem; spectrum=nothing)
  if !isnothing(spectrum)
    E = set_max_truncerror(E, max(max_truncerror(E), truncerror(spectrum)))
  end
  return E
end

function update(
  prob::EigsolveProblem,
  local_state,
  region_iterator;
  outputlevel,
  solver=eigsolve_solver,
  kws...,
)
  eigval, local_state = solver(ψ -> optimal_map(operator(prob), ψ), local_state; kws...)
  prob = set_eigenvalue(prob, eigval)
  if outputlevel >= 2
    @printf(
      "  Region %s: energy = %.12f\n", current_region(region_iterator), eigenvalue(prob)
    )
  end
  return prob, local_state
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

dmrg(args...; kws...) = eigsolve(args...; kws...)
