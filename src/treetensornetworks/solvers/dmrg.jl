function eigsolve_solver(;
  ishermition=true,
  solver_which_eigenvalue=:SR,
  solver_tol=1e-14,
  solver_krylovdim=3,
  solver_maxiter=1,
  solver_verbosity=0,
  kwargs...,
)
  function solver(H, x₀; kws...)
    howmany = 1
    vals, vecs, info = eigsolve(
      H, x₀, howmany, solver_which_eigenvalue;
      ishermitian,
      tol=solver_tol,
      krylovdim=solver_krylovdim,
      maxiter=solver_maxiter,
      verbosity=solver_verbosity,
    )
    x = vecs[1]
    solver_info = (; solver_info=info, energies=vals)
    return x, solver_info
  end
  return solver
end

"""
Overload of `ITensors.dmrg`.
"""
dmrg(A, x₀::AbstractITensorNetwork; kwargs...) = eigsolve(A, x₀; kwargs...)

"""
Overload of `KrylovKit.eigsolve`.
"""
function eigsolve(A, x₀::AbstractITensorNetwork; kwargs...)
  return alternating_update(eigsolve_solver(; kwargs...), A, x₀; kwargs...)
end
