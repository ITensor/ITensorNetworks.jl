function exponentiate_solver(; kwargs...)
  function solver(
    H,
    init;
    ishermitian=true,
    issymmetric=true,
    solver_krylovdim=30,
    solver_maxiter=100,
    solver_outputlevel=0,
    solver_tol=1E-12,
    substep,
    time_step,
    kws...,
  )
    solver_kwargs = (;
      ishermitian,
      issymmetric,
      tol=solver_tol,
      krylovdim=solver_krylovdim,
      maxiter=solver_maxiter,
      verbosity=solver_outputlevel,
      eager=true,
    )

    psi, info = KrylovKit.exponentiate(H, time_step, init; solver_kwargs...)
    return psi, info
  end
  return solver
end

function applyexp_solver(; kwargs...)
  function solver(
    H,
    init;
    tdvp_order,
    solver_krylovdim=30,
    solver_outputlevel=0,
    solver_tol=1E-8,
    substep,
    time_step,
    kws...,
  )
    solver_kwargs = (; maxiter=solver_krylovdim, outputlevel=solver_outputlevel)

    #applyexp tol is absolute, compute from tol_per_unit_time:
    tol = abs(time_step) * tol_per_unit_time
    psi, info = applyexp(H, time_step, init; tol, solver_kwargs..., kws...)
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

function _compute_nsweeps(nsweeps, t, time_step)
  if isinf(t) && isnothing(nsweeps)
    nsweeps = 1
  elseif !isnothing(nsweeps) && time_step != t
    error("Cannot specify both nsweeps and a custom time_step in tdvp")
  elseif isfinite(time_step) && abs(time_step) > 0.0 && isnothing(nsweeps)
    nsweeps = convert(Int, ceil(abs(t / time_step)))
    if !(nsweeps * time_step ≈ t)
      error("Time step $time_step not commensurate with total time t=$t")
    end
  end
  return nsweeps
end

function tdvp(
  solver, H, t::Number, init::AbstractTTN; time_step=t, nsweeps=nothing, order=2, kwargs...
)
  nsweeps = _compute_nsweeps(nsweeps, t, time_step)
  tdvp_order = TDVPOrder(order, Base.Forward)
  return alternating_update(solver, H, init; nsweeps, tdvp_order, time_step, kwargs...)
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
