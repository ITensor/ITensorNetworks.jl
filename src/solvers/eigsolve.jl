
function eigsolve_solver(;
  solver_which_eigenvalue=:SR,   #TODO: settle on pattern to pass solver kwargs
  ishermitian=true,
  solver_tol=1e-14,
  solver_krylovdim=3,
  solver_maxiter=1,
  solver_verbosity=0,
)
  function solver(
    init;
    psi_ref!,
    PH_ref!,
    normalize,
    region,
    sweep_regions,
    sweep_step,
    sweep_kwargs...,
    # slurp solver_kwargs?  #TODO: homogenize how the solver kwargs are passed
  )
    howmany = 1
    which = solver_which_eigenvalue
    vals, vecs, info = eigsolve(
      PH_ref![],
      init,
      howmany,
      which;
      ishermitian,
      tol=solver_tol,
      krylovdim=solver_krylovdim,
      maxiter=solver_maxiter,
      verbosity=solver_verbosity,
    )
    phi = vecs[1]
    return phi, (; solver_info=info, energies=vals)
  end
  return solver
end
