
"""
$(TYPEDSIGNATURES)

Compute a solution x to the linear system:

(a₀ + a₁ * A)*x = b

using starting guess x₀. Leaving a₀, a₁
set to their default values solves the 
system A*x = b.

To adjust the balance between accuracy of solution
and speed of the algorithm, it is recommed to first try
adjusting the `solver_tol` keyword argument descibed below.

Keyword arguments:
  - `ishermitian::Bool=false` - should set to true if the MPO A is Hermitian
  - `solver_krylovdim::Int=30` - max number of Krylov vectors to build on each solver iteration
  - `solver_maxiter::Int=100` - max number outer iterations (restarts) to do in the solver step
  - `solver_tol::Float64=1E-14` - tolerance or error goal of the solver

Overload of `KrylovKit.linsolve`.
"""
function linsolve(
  A::AbstractTTN,
  b::AbstractTTN,
  x₀::AbstractTTN,
  a₀::Number=0,
  a₁::Number=1;
  updater=linsolve_updater,
  nsweeps,  #it makes sense to require this to be defined
  nsites=2,
  (sweep_observer!)=observer(),
  root_vertex=default_root_vertex(init),
  updater_kwargs=(;),
  kwargs...,
)
  updater_kwargs = (; a₀, a₁, updater_kwargs...)
  error("`linsolve` for TTN not yet implemented.")

  sweep_plan = default_sweep_regions(nsites, x0)
  # TODO: Define `itensornetwork_cache`
  # TODO: Define `linsolve_cache`

  P = linsolve_cache(itensornetwork_cache(x₀', A, x₀), itensornetwork_cache(x₀', b))
  return alternating_update(linsolve_updater, P, x₀; sweep_plan, updater_kwargs, kwargs...)
end
