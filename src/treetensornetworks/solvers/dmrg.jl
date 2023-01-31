function eigsolve_solver(; kwargs...)
  function solver(H, t, init; kws...)
    howmany = 1
    which = get(kwargs, :solver_which_eigenvalue, :SR)
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
function dmrg(H, init::AbstractTTN; kwargs...)
  t = Inf # DMRG is TDVP with an infinite timestep and no reverse step
  reverse_step = false
  psi = alternating_update(
    eigsolve_solver(; kwargs...), H, t, init; reverse_step, kwargs...
  )
  return psi
end

"""
Overload of `KrylovKit.eigsolve`.
"""
function eigsolve(H, init::AbstractTTN; kwargs...)
  return dmrg(H, init; kwargs...)
end
