function dmrg_x(operator, init_state::AbstractTTN;;
  nsweeps,
  updater=dmrg_x_updater,
  kwargs...)
    return default_alternating_updates(operator, init_state;
    nsweeps,
    updater,
    kwargs...)
  end