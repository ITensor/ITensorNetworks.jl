function dmrg_x(
  updater,
  operator,
  init::AbstractTTN;
  nsweeps,  #it makes sense to require this to be defined
  nsites=2,
  (sweep_observer!)=observer(),
  root_vertex=default_root_vertex(init),
  updater_kwargs=(;),
  kwargs...,
)
  sweep_plan = dmrg_sweep_plan(nsites, init; root_vertex)

  psi = alternating_update(
    updater, operator, init; nsweeps, sweep_observer!, sweep_plan, updater_kwargs, kwargs...
  )
  return psi
end

function dmrg_x(operator, init::AbstractTTN; updater=dmrg_x_updater, kwargs...)
  return dmrg_x(updater, operator, init; kwargs...)
end

