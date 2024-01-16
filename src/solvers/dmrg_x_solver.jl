function dmrg_x_solver(
  init;
  psi_ref!,
  PH_ref!,
  normalize=nothing,
  region,
  sweep_regions,
  sweep_step,
  half_sweep,
  step_kwargs...,
)
  H = contract(PH_ref![], ITensor(1.0))
  D, U = eigen(H; ishermitian=true)
  u = uniqueind(U, H)
  max_overlap, max_ind = findmax(abs, array(dag(init) * U))
  U_max = U * dag(onehot(u => max_ind))
  # TODO: improve this to return the energy estimate too
  return U_max, NamedTuple()
end
