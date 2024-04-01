using KrylovKit: linsolve

function linsolve_updater(
  init;
  state!,
  projected_operator!,
  outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  region_kwargs,
  ishermitian=false,
  tol=1E-14,
  krylovdim=30,
  maxiter=100,
  verbosity=0,
  a₀,
  a₁,
)
  P = projected_operator![]
  b = dag(only(proj_mps(P)))
  x, info = linsolve(
    P, b, init, a₀, a₁; ishermitian=false, tol, krylovdim, maxiter, verbosity
  )
  return x, (;)
end
