"""
Overload of `ITensors.dmrg`.
"""

function dmrg(operator, init_state;
nsweeps,
updater=eigsolve_updater,
kwargs...)
  return default_alternating_updates(operator, init_state;
  nsweeps,
  updater,
  kwargs...)
end

"""
Overload of `KrylovKit.eigsolve`.
"""
eigsolve(H, init::AbstractTTN; kwargs...) = dmrg(H, init; kwargs...)
