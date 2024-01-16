"""
Overload of `ITensors.dmrg`.
"""
function dmrg(
  H,
  init::AbstractTTN;
  solver_which_eigenvalue=:SR,
  ishermitian=true,
  solver_tol=1e-14,
  solver_krylovdim=3,
  solver_maxiter=1,
  solver_verbosity=0,
  kwargs...,
)
  return alternating_update(
    eigsolve_solver(;
      solver_which_eigenvalue,
      ishermitian,
      solver_tol,
      solver_krylovdim,
      solver_maxiter,
      solver_verbosity,
    ),
    H,
    init;
    reverse_step=false,
    kwargs...,
  )
end

"""
Overload of `KrylovKit.eigsolve`.
"""
eigsolve(H, init::AbstractTTN; kwargs...) = dmrg(H, init; kwargs...)
