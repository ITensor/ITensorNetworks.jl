using KrylovKit: exponentiate

function exponentiate_updater(
  init;
  state!,
  projected_operator!,
  outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  internal_kwargs,
  krylovdim=30,
  maxiter=100,
  verbosity=0,
  tol=1E-12,
  ishermitian=true,
  issymmetric=true,
  eager=true,
)
  (; time_step) = internal_kwargs
  result, exp_info = exponentiate(
    projected_operator![],
    time_step,
    init;
    eager,
    krylovdim,
    maxiter,
    verbosity,
    tol,
    ishermitian,
    issymmetric,
  )
  return result, (; info=exp_info)
end
