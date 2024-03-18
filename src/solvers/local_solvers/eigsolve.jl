function eigsolve_updater(
  init;
  state!,
  projected_operator!,
  outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  internal_kwargs,
  which_eigval=:SR,
  ishermitian=true,
  tol=1e-14,
  krylovdim=3,
  maxiter=1,
  verbosity=0,
  eager=false,
)
  howmany = 1
  vals, vecs, info = eigsolve(
    projected_operator![],
    init,
    howmany,
    which_eigval;
    ishermitian,
    tol,
    krylovdim,
    maxiter,
    verbosity,
    eager,
  )
  return vecs[1], (; info, eigvals=vals)
end
