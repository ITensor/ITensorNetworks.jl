function dmrg_x(PH, init::AbstractTTN; kwargs...)
  psi = alternating_update(ITensorNetworks.dmrg_x_solver, PH, init; kwargs...)
  return psi
end
