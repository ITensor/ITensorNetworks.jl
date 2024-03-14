function dmrg_x_updater(
  init;
  state!,
  projected_operator!,
  outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  internal_kwargs,
)
  #ToDo: Implement this via KrylovKit or similar for better scaling
  H = contract(projected_operator![], ITensor(true))
  D, U = eigen(H; ishermitian=true)
  u = uniqueind(U, H)
  max_overlap, max_ind = findmax(abs, array(dag(init) * U))
  U_max = U * dag(onehot(u => max_ind))
  # TODO: improve this to return the energy estimate too
  return U_max, (;)
end
