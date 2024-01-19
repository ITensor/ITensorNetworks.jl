"""
Overload of `ITensors.dmrg`.
"""

function dmrg_sweep(
  nsites::Int, graph::AbstractGraph; root_vertex=default_root_vertex(graph)
)
  order = 2
  time_step = Inf
  return tdvp_sweep(order, nsites, time_step, graph; root_vertex, reverse_step=false)
end

function dmrg(
  updater,
  H,
  init::AbstractTTN;
  nsweeps,  #it makes sense to require this to be defined
  nsites=2,
  (sweep_observer!)=observer(),
  root_vertex=default_root_vertex(init),
  updater_kwargs=(;),
  kwargs...,
)
  region_updates = dmrg_sweep(nsites, init; root_vertex)

  psi = alternating_update(
    updater, H, init; nsweeps, sweep_observer!, region_updates, updater_kwargs, kwargs...
  )
  return psi
end

function dmrg(H, init::AbstractTTN; updater=eigsolve_updater, kwargs...)
  return dmrg(updater, H, init; kwargs...)
end

"""
Overload of `KrylovKit.eigsolve`.
"""
eigsolve(H, init::AbstractTTN; kwargs...) = dmrg(H, init; kwargs...)
