function exponentiate_solver(; kwargs...)
  function solver(H, t, psi0; kws...)
    solver_kwargs = (;
      ishermitian=get(kwargs, :ishermitian, true),
      issymmetric=get(kwargs, :issymmetric, true),
      tol=get(kwargs, :solver_tol, 1E-12),
      krylovdim=get(kwargs, :solver_krylovdim, 30),
      maxiter=get(kwargs, :solver_maxiter, 100),
      verbosity=get(kwargs, :solver_outputlevel, 0),
      eager=true,
    )
    psi, info = exponentiate(H, t, psi0; solver_kwargs...)
    return psi, info
  end
  return solver
end

function applyexp_solver(; kwargs...)
  function solver(H, t, psi0; kws...)
    tol_per_unit_time = get(kwargs, :solver_tol, 1E-8)
    solver_kwargs = (;
      maxiter=get(kwargs, :solver_krylovdim, 30),
      outputlevel=get(kwargs, :solver_outputlevel, 0),
    )
    #applyexp tol is absolute, compute from tol_per_unit_time:
    tol = abs(t) * tol_per_unit_time
    psi, info = applyexp(H, t, psi0; tol, solver_kwargs..., kws...)
    return psi, info
  end
  return solver
end

function tdvp_solver(; kwargs...)
  solver_backend = get(kwargs, :solver_backend, "applyexp")
  if solver_backend == "applyexp"
    return applyexp_solver(; kwargs...)
  elseif solver_backend == "exponentiate"
    return exponentiate_solver(; kwargs...)
  else
    error(
      "solver_backend=$solver_backend not recognized (options are \"applyexp\" or \"exponentiate\")",
    )
  end
end

function tdvp(H, t::Number, psi0::MPS; kwargs...)
  return tdvp(tdvp_solver(; kwargs...), H, t, psi0; kwargs...)
end

function tdvp(t::Number, H, psi0::MPS; kwargs...)
  return tdvp(H, t, psi0; kwargs...)
end

function tdvp(H, psi0::MPS, t::Number; kwargs...)
  return tdvp(H, t, psi0; kwargs...)
end
