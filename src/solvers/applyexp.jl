function applyexp_solver()
    function solver(
      init;
      psi_ref!,
      PH_ref!,
      region,
      sweep_regions,
      sweep_step,
      solver_krylovdim=30,
      solver_outputlevel=0,
      solver_tol=1E-8,
      substep,
      time_step,
      normalize,
      )
      H=PH_ref![]
      #applyexp tol is absolute, compute from tol_per_unit_time:
      tol = abs(time_step) * tol_per_unit_time
      psi, exp_info = applyexp(
        H, time_step, init; tol, maxiter=solver_krylovdim, outputlevel=solver_outputlevel
      )
      return psi, (; info=exp_info)
    end
    return solver
end
  

