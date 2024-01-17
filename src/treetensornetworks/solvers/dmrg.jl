"""
Overload of `ITensors.dmrg`.
"""


function dmrg_sweep(
  nsite::Int,
  graph::AbstractGraph;
  root_vertex=default_root_vertex(graph),
)
  return tdvp_sweep(2,nsite,Inf,graph;root_vertex,reverse_step=false)
end


function default_sweep_regions(nsite, graph::AbstractGraph; kwargs...)  ###move this to a different file, algorithmic level idea
  return vcat(
    [
      half_sweep(
        direction(half),
        graph,
        make_region;
        nsite,
        region_args=(; substep=half),
        kwargs...,
      ) for half in 1:2
    ]...,
  )
end

function dmrg(
  updater,
  H,
  init::AbstractTTN;
  nsweeps,  #it makes sense to require this to be defined
  nsite=2,
  (sweep_observer!)=observer(),
  root_vertex=default_root_vertex(init),
  updater_kwargs=NamedTuple(;),
  kwargs...,
  )
  region_updates = dmrg_sweep(nsite,init; root_vertex)

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


