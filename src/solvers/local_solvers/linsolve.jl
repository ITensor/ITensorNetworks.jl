function linsolve_updater(
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
    ishermitian=false, tol=1E-14, krylovdim=30, maxiter=100, verbosity=0, a₀, a₁
  )
  updater_kwargs = merge(default_updater_kwargs, updater_kwargs)
  P = projected_operator![]
  (; a₀, a₁) = updater_kwargs
  updater_kwargs = Base.structdiff(updater_kwargs, (; a₀=nothing, a₁=nothing))
  b = dag(only(proj_mps(P)))
  x, info = KrylovKit.linsolve(P, b, init, a₀, a₁; updater_kwargs...)
  return x, (;)
end
