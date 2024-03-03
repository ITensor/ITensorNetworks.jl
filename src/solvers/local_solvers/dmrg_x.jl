function dmrg_x_updater(
  init;
  state!,
  projected_operator!,
  outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  region_kwargs,
  updater_kwargs,
)
  # this updater does not seem to accept any kwargs?
  default_updater_kwargs = (;)
  updater_kwargs = merge(default_updater_kwargs, updater_kwargs)
  H = contract(projected_operator![], ITensor(true))
  D, U = eigen(H; ishermitian=true)
  u = uniqueind(U, H)
  max_overlap, max_ind = findmax(abs, array(dag(init) * U))
  U_max = U * dag(onehot(u => max_ind))
  # TODO: improve this to return the energy estimate too
  return U_max, (;)
end
