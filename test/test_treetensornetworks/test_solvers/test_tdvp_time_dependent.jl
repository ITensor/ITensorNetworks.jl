using DifferentialEquations
using ITensors
using ITensorNetworks
using KrylovKit: exponentiate
using LinearAlgebra
using Test

const ttn_solvers_examples_dir = joinpath(
  pkgdir(ITensorNetworks), "examples", "treetensornetworks", "solvers"
)

include(joinpath(ttn_solvers_examples_dir, "03_models.jl"))
include(joinpath(ttn_solvers_examples_dir, "03_solvers.jl"))

# Functions need to be defined in global scope (outside
# of the @testset macro)

ω₁ = 0.1
ω₂ = 0.2

ode_alg = Tsit5()
ode_kwargs = (; reltol=1e-8, abstol=1e-8)

ω⃗ = [ω₁, ω₂]
f⃗ = [t -> cos(ω * t) for ω in ω⃗]

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

krylov_kwargs = (; tol=1e-8, eager=true)

function krylov_solver(H⃗₀, time_step, ψ₀; kwargs...)
  return krylov_solver(
    -im * TimeDependentSum(f⃗, H⃗₀), time_step, ψ₀; krylov_kwargs..., kwargs...
  )
end

@testset "MPS: Time dependent Hamiltonian" begin
  n = 4
  J₁ = 1.0
  J₂ = 0.1

  time_step = 0.1
  time_stop = 1.0

  nsite = 2
  maxdim = 100
  cutoff = 1e-8

  s = siteinds("S=1/2", n)
  ℋ₁₀ = ITensorNetworks.heisenberg(n; J1=J₁, J2=0.0)
  ℋ₂₀ = ITensorNetworks.heisenberg(n; J1=0.0, J2=J₂)
  ℋ⃗₀ = [ℋ₁₀, ℋ₂₀]
  H⃗₀ = [mpo(ℋ₀, s) for ℋ₀ in ℋ⃗₀]

  ψ₀ = complex(mps(s; states=(j -> isodd(j) ? "↑" : "↓")))

  ψₜ_ode = tdvp(ode_solver, H⃗₀, time_stop, ψ₀; time_step, maxdim, cutoff, nsite)

  ψₜ_krylov = tdvp(krylov_solver, H⃗₀, time_stop, ψ₀; time_step, cutoff, nsite)

  ψₜ_full, _ = ode_solver(contract.(H⃗₀), time_stop, contract(ψ₀))

  @test norm(ψ₀) ≈ 1
  @test norm(ψₜ_ode) ≈ 1
  @test norm(ψₜ_krylov) ≈ 1
  @test norm(ψₜ_full) ≈ 1

  ode_err = norm(contract(ψₜ_ode) - ψₜ_full)
  krylov_err = norm(contract(ψₜ_krylov) - ψₜ_full)

  @test krylov_err > ode_err
  @test ode_err < 1e-3
  @test krylov_err < 1e-3
end

@testset "TTN: Time dependent Hamiltonian" begin
  tooth_lengths = fill(2, 3)
  root_vertex = (3, 2)
  c = named_comb_tree(tooth_lengths)
  s = siteinds("S=1/2", c)

  J₁ = 1.0
  J₂ = 0.1

  time_step = 0.1
  time_stop = 1.0

  nsite = 2
  maxdim = 100
  cutoff = 1e-8

  s = siteinds("S=1/2", c)
  ℋ₁₀ = ITensorNetworks.heisenberg(c; J1=J₁, J2=0.0)
  ℋ₂₀ = ITensorNetworks.heisenberg(c; J1=0.0, J2=J₂)
  ℋ⃗₀ = [ℋ₁₀, ℋ₂₀]
  H⃗₀ = [TTN(ℋ₀, s) for ℋ₀ in ℋ⃗₀]

  ψ₀ = TTN(ComplexF64, s, v -> iseven(sum(isodd.(v))) ? "↑" : "↓")

  ψₜ_ode = tdvp(ode_solver, H⃗₀, time_stop, ψ₀; time_step, maxdim, cutoff, nsite)

  ψₜ_krylov = tdvp(krylov_solver, H⃗₀, time_stop, ψ₀; time_step, cutoff, nsite)

  ψₜ_full, _ = ode_solver(contract.(H⃗₀), time_stop, contract(ψ₀))

  @test norm(ψ₀) ≈ 1
  @test norm(ψₜ_ode) ≈ 1
  @test norm(ψₜ_krylov) ≈ 1
  @test norm(ψₜ_full) ≈ 1

  ode_err = norm(contract(ψₜ_ode) - ψₜ_full)
  krylov_err = norm(contract(ψₜ_krylov) - ψₜ_full)

  @test krylov_err > ode_err
  @test ode_err < 1e-3
  @test krylov_err < 1e-2
end

nothing
