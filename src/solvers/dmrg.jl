using ITensorMPS: ITensorMPS, dmrg
using KrylovKit: KrylovKit

"""
Overload of `ITensors.ITensorMPS.dmrg`.
"""
function ITensorMPS.dmrg(
  operator,
  init_state;
  nsweeps,
  nsites=2,
  updater=eigsolve_updater,
  (region_observer!)=nothing,
  kwargs...,
)
  eigvals_ref = Ref{Any}()
  region_observer! = compose_observers(
    region_observer!, ValuesObserver((; eigvals=eigvals_ref))
  )
  state = alternating_update(
    operator, init_state; nsweeps, nsites, updater, region_observer!, kwargs...
  )
  eigval = first(eigvals_ref[])
  return eigval, state
end

"""
Overload of `KrylovKit.eigsolve`.
"""
KrylovKit.eigsolve(H, init::AbstractTTN; kwargs...) = dmrg(H, init; kwargs...)
