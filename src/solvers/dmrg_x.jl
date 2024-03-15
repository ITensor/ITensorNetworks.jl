function dmrg_x(operator, init_state::AbstractTTN;
  nsweeps,
  nsites=2,
  updater=dmrg_x_updater,
  kwargs...)
    return alternating_update(operator, init_state;
    nsweeps,
    nsites,
    updater,
    kwargs...)
  end