function dmrg_x_solver(
  PH, init; normalize=nothing, region=nothing, half_sweep=nothing, reverse_step=nothing
)
  H = contract(PH, ITensor(1.0))
  D, U = eigen(H; ishermitian=true)
  u = uniqueind(U, H)
  max_overlap, max_ind = findmax(abs, array(dag(init) * U))
  U_max = U * dag(onehot(u => max_ind))
  # TODO: improve this to return the energy estimate too
  return U_max, NamedTuple()
end

function dmrg_x(PH, init::AbstractTTN; kwargs...)
  psi = alternating_update(dmrg_x_solver, PH, init; kwargs...)
  return psi
end
