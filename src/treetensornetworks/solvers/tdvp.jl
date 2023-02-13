function exponentiate_solver(; kwargs...)
  function solver(H, t, init; kws...)
    solver_kwargs = (;
      ishermitian=get(kwargs, :ishermitian, true),
      issymmetric=get(kwargs, :issymmetric, true),
      tol=get(kwargs, :solver_tol, 1E-12),
      krylovdim=get(kwargs, :solver_krylovdim, 30),
      maxiter=get(kwargs, :solver_maxiter, 100),
      verbosity=get(kwargs, :solver_outputlevel, 0),
      eager=true,
    )
    psi, info = exponentiate(H, t, init; solver_kwargs...)
    return psi, info
  end
  return solver
end

function applyexp_solver(; kwargs...)
  function solver(H, t, init; kws...)
    tol_per_unit_time = get(kwargs, :solver_tol, 1E-8)
    solver_kwargs = (;
      maxiter=get(kwargs, :solver_krylovdim, 30),
      outputlevel=get(kwargs, :solver_outputlevel, 0),
    )
    #applyexp tol is absolute, compute from tol_per_unit_time:
    tol = abs(t) * tol_per_unit_time
    psi, info = applyexp(H, t, init; tol, solver_kwargs..., kws...)
    return psi, info
  end
  return solver
end

function tdvp_solver(; solver_backend="exponentiate", kwargs...)
  if solver_backend == "exponentiate"
    return exponentiate_solver(; kwargs...)
  elseif solver_backend == "applyexp"
    return applyexp_solver(; kwargs...)
  else
    error(
      "solver_backend=$solver_backend not recognized (options are \"applyexp\" or \"exponentiate\")",
    )
  end
end

function tdvp(solver, H, t::Number, init::AbstractTTN; kwargs...)
  return alternating_update(solver, H, t, init; kwargs...)
end

"""
    tdvp(H::TTN, t::Number, psi0::TTN; kwargs...)

Use the time dependent variational principle (TDVP) algorithm
to compute `exp(H*t)*psi0` using an efficient algorithm based
on alternating optimization of the state tensors and local Krylov
exponentiation of H.
                    
Returns:
* `psi` - time-evolved state

Optional keyword arguments:
* `outputlevel::Int = 1` - larger outputlevel values resulting in printing more information and 0 means no output
* `observer` - object implementing the [Observer](@ref observer) interface which can perform measurements and stop early
* `write_when_maxdim_exceeds::Int` - when the allowed maxdim exceeds this value, begin saving tensors to disk to free memory in large calculations
"""
function tdvp(H, t::Number, init::AbstractTTN; kwargs...)
  return tdvp(tdvp_solver(; kwargs...), H, t, init; kwargs...)
end
