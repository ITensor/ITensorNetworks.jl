using ITensors.ITensorMPS: ITensorMPS, dmrg
using KrylovKit: KrylovKit

struct ComposedObservers{Observers<:Tuple}
  observers::Observers
end
compose_observers(observers...) = ComposedObservers(observers)

function eigval_observer(; kwargs...)
  @show kwargs
  error()
end

"""
Overload of `ITensors.ITensorMPS.dmrg`.
"""
function ITensorMPS.dmrg(
  operator, init_state; nsweeps, nsites=2, updater=eigsolve_updater, (sweep_observer!)=nothing, kwargs...
)
  sweep_observer! = compose_observers(sweep_observer!, eigval_observer)
  state = alternating_update(operator, init_state; nsweeps, nsites, updater, sweep_observer!, kwargs...)
  return eigval, state
end

"""
Overload of `KrylovKit.eigsolve`.
"""
KrylovKit.eigsolve(H, init::AbstractTTN; kwargs...) = dmrg(H, init; kwargs...)
