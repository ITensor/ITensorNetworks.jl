function exponentiate_solver()
  function solver(
    init;
    psi_ref!,
    PH_ref!,
    ishermitian=true,
    issymmetric=true,
    region,
    sweep_regions,
    sweep_step,
    solver_krylovdim=30,
    solver_maxiter=100,
    solver_outputlevel=0,
    solver_tol=1E-12,
    substep,
    normalize,
    time_step,
  )
    #H=copy(PH_ref![])
    H = PH_ref![] ###since we are not changing H we don't need the copy
    # let's test whether given region and sweep regions we can find out what the previous and next region were
    # this will be needed in subspace expansion
    region_ind = sweep_step
    next_region =
      region_ind == length(sweep_regions) ? nothing : first(sweep_regions[region_ind + 1])
    previous_region = region_ind == 1 ? nothing : first(sweep_regions[region_ind - 1])

    phi, exp_info = KrylovKit.exponentiate(
      H,
      time_step,
      init;
      ishermitian,
      issymmetric,
      tol=solver_tol,
      krylovdim=solver_krylovdim,
      maxiter=solver_maxiter,
      verbosity=solver_outputlevel,
      eager=true,
    )
    return phi, (; info=exp_info)
  end
  return solver
end
