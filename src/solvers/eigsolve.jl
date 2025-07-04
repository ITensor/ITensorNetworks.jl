using Printf: @printf

@kwdef mutable struct EigsolveProblem{State,Operator}
  operator::Operator
  state::State
  eigenvalue::Number = Inf
end

eigenvalue(E::EigsolveProblem) = E.eigenvalue
ITensorNetworks.state(E::EigsolveProblem) = E.state
operator(E::EigsolveProblem) = E.operator

function set_operator(E::EigsolveProblem, operator)
  EigsolveProblem(operator, E.state, E.eigenvalue)
end
function set_eigenvalue(E::EigsolveProblem, eigenvalue)
  EigsolveProblem(E.operator, E.state, eigenvalue)
end
set_state(E::EigsolveProblem, state) = EigsolveProblem(E.operator, state, E.eigenvalue)

function updater(
  prob::EigsolveProblem,
  local_state,
  region_iterator;
  outputlevel,
  solver=eigsolve_solver,
  kws...,
)
  eigval, local_state = solver(ψ->optimal_map(operator(prob), ψ), local_state; kws...)
  prob = set_eigenvalue(prob, eigval)
  if outputlevel >= 2
    @printf(
      "  Region %s: energy = %.12f\n", current_region(region_iterator), eigenvalue(prob)
    )
  end
  return prob, local_state
end

function eigsolve_sweep_printer(region_iterator; outputlevel, sweep, nsweeps, kws...)
  if outputlevel >= 1
    if nsweeps >= 10
      @printf("After sweep %02d/%d ", sweep, nsweeps)
    else
      @printf("After sweep %d/%d ", sweep, nsweeps)
    end
    prob = problem(region_iterator)
    @printf("eigenvalue=%.12f ", eigenvalue(prob))
    @printf("maxlinkdim=%d", maxlinkdim(state(prob)))
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
  extracter_kwargs=(;),
  updater_kwargs=(;),
  inserter_kwargs=(;),
  sweep_printer=eigsolve_sweep_printer,
  kws...,
)
  init_prob = EigsolveProblem(;
    state=align_indices(init_state), operator=ProjTTN(align_indices(operator))
  )
  sweep_iter = sweep_iterator(
    init_prob,
    nsweeps;
    nsites,
    outputlevel,
    extracter_kwargs,
    updater_kwargs,
    inserter_kwargs,
  )
  prob = sweep_solve(sweep_iter; outputlevel, sweep_printer, kws...)
  return eigenvalue(prob), state(prob)
end

dmrg(args...; kws...) = eigsolve(args...; kws...)
