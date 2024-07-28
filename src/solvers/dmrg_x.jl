function dmrg_x(
  operator,
  init_state::AbstractTTN;
  nsweeps,
  nsites=2,
  updater=dmrg_x_updater,
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
  eigval = only(eigvals_ref[])
  return eigval, state
end
