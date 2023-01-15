function dmrg_x_solver(PH, t, psi0; kwargs...)
  H = contract(PH, ITensor(1.0))
  D, U = eigen(H; ishermitian=true)
  u = uniqueind(U, H)
  max_overlap, max_ind = findmax(abs, array(psi0 * U))
  U_max = U * dag(onehot(u => max_ind))
  return U_max, nothing
end

function dmrg_x(PH, psi0::IsTreeState; reverse_step=false, kwargs...)
  t = Inf
  psi = tdvp(dmrg_x_solver, PH, t, psi0; reverse_step, kwargs...)
  return psi
end
