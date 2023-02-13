function dmrg_x_solver(PH, t, init; kwargs...)
  H = contract(PH, ITensor(1.0))
  D, U = eigen(H; ishermitian=true)
  u = uniqueind(U, H)
  max_overlap, max_ind = findmax(abs, array(dag(init) * U))
  U_max = U * dag(onehot(u => max_ind))
  return U_max, nothing
end

function dmrg_x(PH, init::AbstractTTN; reverse_step=false, kwargs...)
  t = Inf
  psi = alternating_update(dmrg_x_solver, PH, t, init; reverse_step, kwargs...)
  return psi
end
