function eigsolve_updater(
  init;
  state!,
  projected_operator!,
  outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  region_kwargs,
  updater_kwargs,
)
  default_updater_kwargs = (;
    which_eigval=:SR,
    ishermitian=true,
    tol=1e-14,
    krylovdim=3,
    maxiter=1,
    verbosity=0,
    eager=false,
  )
  updater_kwargs = merge(default_updater_kwargs, updater_kwargs)  #last collection has precedence
  howmany = 1
  (; which_eigval) = updater_kwargs
  updater_kwargs = Base.structdiff(updater_kwargs, (; which_eigval=nothing))
  vals, vecs, info = eigsolve(
    projected_operator![], init, howmany, which_eigval; updater_kwargs...
  )
  return vecs[1], (; info, eigvals=vals)
end

function _pop_which_eigenvalue(; which_eigenvalue, kwargs...)
  return which_eigenvalue, NamedTuple(kwargs)
end
