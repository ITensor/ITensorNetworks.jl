function eigsolve_solver(; solver_which_eigenvalue=:SR, kwargs...)
  function solver(H, init; kws...)
    howmany = 1
    which = solver_which_eigenvalue
    solver_kwargs = (;
      ishermitian=get(kwargs, :ishermitian, true),
      tol=get(kwargs, :solver_tol, 1E-14),
      krylovdim=get(kwargs, :solver_krylovdim, 3),
      maxiter=get(kwargs, :solver_maxiter, 1),
      verbosity=get(kwargs, :solver_verbosity, 0),
    )
    vals, vecs, info = eigsolve(H, init, howmany, which; solver_kwargs...)
    psi = vecs[1]
    return psi, info
  end
  return solver
end

"""
Overload of `ITensors.dmrg`.
"""
function dmrg(H, init::AbstractTTN; nsite=2, kwargs...)
  # TODO: move this logic inside alternating_update
  sweep_pattern = nothing
  if nsite == 1
    sweep_pattern = one_site_sweep(H)
  elseif nsite == 2
    sweep_pattern = two_site_sweep(H)
  else
    error("nsite=$nsite not supported in DMRG")
  end
  psi = alternating_update(eigsolve_solver(; kwargs...), H, init; sweep_pattern, kwargs...)
  return psi
end

"""
Overload of `KrylovKit.eigsolve`.
"""
eigsolve(H, init::AbstractTTN; kwargs...) = dmrg(H, init; kwargs...)
