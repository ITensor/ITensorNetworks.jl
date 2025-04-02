import ITensorNetworks as itn
import ITensors as it
using Printf

@kwdef mutable struct EigsolveProblem{State,Operator}
  state::State
  operator::Operator
  eigenvalue::Number = Inf
end

eigenvalue(E::EigsolveProblem) = E.eigenvalue
state(E::EigsolveProblem) = E.state
operator(E::EigsolveProblem) = E.operator

function set(
  E::EigsolveProblem; state=state(E), operator=operator(E), eigenvalue=eigenvalue(E)
)
  return EigsolveProblem(; state, operator, eigenvalue)
end

function updater!(E::EigsolveProblem, local_tensor, region; outputlevel, kws...)
  E.eigenvalue, local_tensor = eigsolve_updater(operator(E), local_tensor; kws...)
  if outputlevel >= 2
    @printf("  Region %s: energy = %.12f\n", region, eigenvalue(E))
  end
  return local_tensor
end

function eigsolve(
  H, init_state; nsweeps, nsites=2, outputlevel=0, updater_kwargs=(;), inserter_kwargs=(;), kws...
)
  init_prob = EigsolveProblem(; state=copy(init_state), operator=itn.ProjTTN(H))
  common_sweep_kwargs = (; nsites, outputlevel, updater_kwargs, inserter_kwargs)
  kwargs_array = [(; common_sweep_kwargs..., sweep=s) for s in 1:nsweeps]
  sweep_iter = sweep_iterator(init_prob, kwargs_array)
  converged_prob = alternating_update(sweep_iter; outputlevel, kws...)
  return eigenvalue(converged_prob), state(converged_prob)
end

dmrg(args...; kws...) = eigsolve(args...; kws...)
