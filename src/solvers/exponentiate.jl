function exponentiate_updater(
  init;
  state!,
  projected_operator!,
  outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  updater_kwargs,
  internal_kwargs,
)
  default_updater_kwargs = (;
    krylovdim=30,
    maxiter=100,
    verbosity=0,
    tol=1E-12,
    ishermitian=true,
    issymmetric=true,
    eager=true,
  )
  # set defaults for unspecified kwargs
  updater_kwargs = merge(default_updater_kwargs, updater_kwargs)  #last collection has precedence
  (; time_step) = internal_kwargs
  result, exp_info = exponentiate(projected_operator![], time_step, init; updater_kwargs...)
  return result, (; info=exp_info)
end
