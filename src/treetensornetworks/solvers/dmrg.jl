function eigsolve_solver(;
  solver_which_eigenvalue=:SR,
  ishermitian=true,
  solver_tol=1e-14,
  solver_krylovdim=3,
  solver_maxiter=1,
  solver_verbosity=0,
)
  function solver(H, init; normalize=nothing, region=nothing, half_sweep=nothing)
    howmany = 1
    which = solver_which_eigenvalue
    vals, vecs, info = eigsolve(
      H,
      init,
      howmany,
      which;
      ishermitian,
      tol=solver_tol,
      krylovdim=solver_krylovdim,
      maxiter=solver_maxiter,
      verbosity=solver_verbosity,
    )
    psi = vecs[1]
    return psi, (; solver_info=info, energies=vals)
  end
  return solver
end

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
    kwargs...,
  )
end

"""
Overload of `KrylovKit.eigsolve`.
"""
eigsolve(H, init::AbstractTTN; kwargs...) = dmrg(H, init; kwargs...)
