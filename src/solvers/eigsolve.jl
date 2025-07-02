using Printf: @printf
import ConstructionBase: setproperties

@kwdef mutable struct EigsolveProblem{State,Operator}
  state::State
  operator::Operator
  eigenvalue::Number = Inf
end

eigenvalue(E::EigsolveProblem) = E.eigenvalue
ITensorNetworks.state(E::EigsolveProblem) = E.state
operator(E::EigsolveProblem) = E.operator

function updater(
  E::EigsolveProblem,
  local_state,
  region_iterator;
  outputlevel,
  solver=eigsolve_solver,
  kws...,
)
  eigval, local_state = solver(ψ->optimal_map(operator(E), ψ), local_state; kws...)
  E = setproperties(E; eigenvalue=eigval)
  if outputlevel >= 2
    @printf("  Region %s: energy = %.12f\n", current_region(region_iterator), eigenvalue(E))
  end
  return E, local_state
end

function eigsolve_sweep_printer(region_iterator; outputlevel, sweep, nsweeps, kws...)
  if outputlevel >= 1
    if nsweeps >= 10
      @printf("After sweep %02d/%d ", sweep, nsweeps)
    else
      @printf("After sweep %d/%d ", sweep, nsweeps)
    end
    E = problem(region_iterator)
    @printf("eigenvalue=%.12f ", eigenvalue(E))
    @printf("maxlinkdim=%d", maxlinkdim(state(E)))
    println()
    flush(stdout)
  end
end

function eigsolve(
  init_prob;
  nsweeps,
  nsites=1,
  outputlevel=0,
  extracter_kwargs=(;),
  updater_kwargs=(;),
  inserter_kwargs=(;),
  sweep_printer=eigsolve_sweep_printer,
  kws...,
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

function eigsolve(operator, init_state; kws...)
  init_prob = EigsolveProblem(;
    state=permute_indices(init_state), operator=ProjTTN(permute_indices(operator))
  )
  return eigsolve(init_prob; kws...)
end

dmrg(args...; kws...) = eigsolve(args...; kws...)
