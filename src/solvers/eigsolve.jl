function eigsolve_updater(
  init;
  state!,
  projected_operator!,
  outputlevel,
  which_sweep,
  region_updates,
  which_region_update,
  region_kwargs,
  updater_kwargs,
)
  default_updater_kwargs = (;
    which_eigenvalue=:SR,
    ishermitian=true,
    tol=1e-14,
    krylovdim=3,
    maxiter=1,
    outputlevel=0,
    eager=false,
  )
  updater_kwargs = merge(default_updater_kwargs, updater_kwargs)  #last collection has precedence
  howmany = 1
  which = updater_kwargs.which_eigenvalue
  vals, vecs, info = eigsolve(
    projected_operator![],
    init,
    howmany,
    which;
    ishermitian=updater_kwargs.ishermitian,
    tol=updater_kwargs.tol,
    krylovdim=updater_kwargs.krylovdim,
    maxiter=updater_kwargs.maxiter,
    verbosity=updater_kwargs.outputlevel,
    eager=updater_kwargs.eager,
  )
  return vecs[1], (; info, eigvals=vals)
end
