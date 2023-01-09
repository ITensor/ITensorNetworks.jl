using DifferentialEquations
using ITensors
using ITensorNetworks
using KrylovKit
using LinearAlgebra
using Random

Random.seed!(1234)

# Define the time-independent model
include("03_models.jl")

# Define the solvers needed for TDVP
include("03_solvers.jl")

# Time dependent Hamiltonian is:
# H(t) = H₁(t) + H₂(t) + …
#      = f₁(t) H₁(0) + f₂(t) H₂(0) + …
#      = cos(ω₁t) H₁(0) + cos(ω₂t) H₂(0) + …

# Number of sites
n = 6

# How much information to output from TDVP
# Set to 2 to get information about each bond/site
# evolution, and 3 to get information about the
# solver.
outputlevel = 1

# Frequency of time dependent terms
ω₁ = 0.1
ω₂ = 0.2

# Nearest and next-nearest neighbor
# Heisenberg couplings.
J₁ = 1.0
J₂ = 1.0

time_step = 0.1
time_stop = 1.0

# nsite-update TDVP
nsite = 2

# Starting state bond/link dimension.
# A product state starting state can
# cause issues for TDVP without
# subspace expansion.
start_linkdim = 4

# TDVP truncation parameters
maxdim = 100
cutoff = 1e-8

tol = 1e-15

# ODE solver parameters
ode_alg = Tsit5()
ode_kwargs = (; reltol=tol, abstol=tol)

# Krylov solver parameters
krylov_kwargs = (; tol=tol, eager=true)

@show n
@show ω₁, ω₂
@show J₁, J₂
@show maxdim, cutoff, nsite
@show start_linkdim
@show time_step, time_stop
@show ode_alg
@show ode_kwargs
@show krylov_kwargs

ω⃗ = [ω₁, ω₂]
f⃗ = [t -> cos(ω * t) for ω in ω⃗]

# H₀ = H(0) = H₁(0) + H₂(0) + …
ℋ₁₀ = heisenberg(n; J=J₁, J2=0.0)
ℋ₂₀ = heisenberg(n; J=0.0, J2=J₂)
ℋ⃗₀ = [ℋ₁₀, ℋ₂₀]

s = siteinds("S=1/2", n)

H⃗₀ = [MPO(ℋ₀, s) for ℋ₀ in ℋ⃗₀]

# Initial state, ψ₀ = ψ(0)
# Initialize as complex since that is what DifferentialEquations.jl
# expects.
ψ₀ = complex.(randomMPS(s, j -> isodd(j) ? "↑" : "↓"; linkdims=start_linkdim))

@show norm(ψ₀)

println()
println("#"^100)
println("Running TDVP with ODE solver")
println("#"^100)
println()

function ode_solver(H⃗₀, time_step, ψ₀; kwargs...)
  return ode_solver(
    -im * TimeDependentSum(f⃗, H⃗₀),
    time_step,
    ψ₀;
    solver_alg=ode_alg,
    ode_kwargs...,
    kwargs...,
  )
end

ψₜ_ode = tdvp(ode_solver, H⃗₀, time_stop, ψ₀; time_step, maxdim, cutoff, nsite, outputlevel)

println()
println("Finished running TDVP with ODE solver")
println()

println()
println("#"^100)
println("Running TDVP with Krylov solver")
println("#"^100)
println()

function krylov_solver(H⃗₀, time_step, ψ₀; kwargs...)
  return krylov_solver(
    -im * TimeDependentSum(f⃗, H⃗₀), time_step, ψ₀; krylov_kwargs..., kwargs...
  )
end

ψₜ_krylov = tdvp(krylov_solver, H⃗₀, time_stop, ψ₀; time_step, cutoff, nsite, outputlevel)

println()
println("Finished running TDVP with Krylov solver")
println()

println()
println("#"^100)
println("Running full state evolution with ODE solver")
println("#"^100)
println()

@disable_warn_order begin
  ψₜ_full, _ = ode_solver(prod.(H⃗₀), time_stop, prod(ψ₀); outputlevel)
end

println()
println("Finished full state evolution with ODE solver")
println()

@show norm(ψₜ_ode)
@show norm(ψₜ_krylov)
@show norm(ψₜ_full)

@show 1 - abs(inner(prod(ψₜ_ode), ψₜ_full))
@show 1 - abs(inner(prod(ψₜ_krylov), ψₜ_full))
