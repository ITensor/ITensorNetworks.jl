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
ode_updater_kwargs=(;f=f⃗,solver_alg=ode_alg,ode_kwargs)

function ode_updater(
  init;
  state!,
  projected_operator!,
  outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  region_kwargs,
  updater_kwargs,
)
    time_step=region_kwargs.time_step
    f⃗=updater_kwargs.f
    ode_kwargs=updater_kwargs.ode_kwargs
    solver_alg=updater_kwargs.solver_alg
    H⃗₀=projected_operator![]
    result, info = ode_solver(
    -im * TimeDependentSum(f⃗, H⃗₀), time_step, init; solver_alg, ode_kwargs...
  )
  return result, (; info)
end

function tdvp_ode_solver(H⃗₀, ψ₀; time_step, kwargs...)
  psi_t, info = ode_solver(
    -im * TimeDependentSum(f⃗, H⃗₀), time_step, ψ₀; solver_alg=ode_alg, ode_kwargs...
  )
  return psi_t, (; info)
end


krylov_kwargs = (; tol=1e-8, eager=true)
krylov_updater_kwargs=(;f=f⃗,krylov_kwargs)

function krylov_solver(H⃗₀, ψ₀; time_step, ishermitian=false, issymmetric=false, kwargs...)
  psi_t, info = krylov_solver(
    -im * TimeDependentSum(f⃗, H⃗₀),
    time_step,
    ψ₀;
    krylov_kwargs...,
    ishermitian,
    issymmetric,
  )
  return psi_t, (; info)
end

function krylov_updater(
  init;
  state!,
  projected_operator!,
  outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  region_kwargs,
  updater_kwargs,
)
  default_updater_kwargs = (;
    ishermitian=false,
    issymmetric=false,
  )

  updater_kwargs = merge(default_updater_kwargs, updater_kwargs)  #last collection has precedenc
    time_step=region_kwargs.time_step
    f⃗=updater_kwargs.f
    krylov_kwargs=updater_kwargs.krylov_kwargs
    ishermitian=updater_kwargs.ishermitian
    issymmetric=updater_kwargs.issymmetric
    H⃗₀=projected_operator![]

    result, info = krylov_solver(
    -im * TimeDependentSum(f⃗, H⃗₀),
    time_step,
    init;
    krylov_kwargs...,
    ishermitian,
    issymmetric,
  )
  return result, (; info)
end


@testset "MPS: Time dependent Hamiltonian" begin
  n = 4
  J₁ = 1.0
  J₂ = 0.1

  time_step = 0.1
  time_total = 1.0

  nsites = 2
  maxdim = 100
  cutoff = 1e-8

  s = siteinds("S=1/2", n)
  ℋ₁₀ = ITensorNetworks.heisenberg(n; J1=J₁, J2=0.0)
  ℋ₂₀ = ITensorNetworks.heisenberg(n; J1=0.0, J2=J₂)
  ℋ⃗₀ = [ℋ₁₀, ℋ₂₀]
  H⃗₀ = [mpo(ℋ₀, s) for ℋ₀ in ℋ⃗₀]

  ψ₀ = complex(mps(s; states=(j -> isodd(j) ? "↑" : "↓")))

  ψₜ_ode = tdvp(ode_updater, H⃗₀, time_total, ψ₀; time_step, maxdim, cutoff, nsites, updater_kwargs=ode_updater_kwargs)

  ψₜ_krylov = tdvp(krylov_updater, H⃗₀, time_total, ψ₀; time_step, cutoff, nsites, updater_kwargs=krylov_updater_kwargs)

  ψₜ_full, _ = tdvp_ode_solver(contract.(H⃗₀), contract(ψ₀); time_step=time_total)

  @test norm(ψ₀) ≈ 1
  @test norm(ψₜ_ode) ≈ 1
  @test norm(ψₜ_krylov) ≈ 1
  @test norm(ψₜ_full) ≈ 1

  ode_err = norm(contract(ψₜ_ode) - ψₜ_full)
  krylov_err = norm(contract(ψₜ_krylov) - ψₜ_full)

  @test krylov_err > ode_err
  @test ode_err < 1e-2
  @test krylov_err < 1e-2
end


@testset "TTN: Time dependent Hamiltonian" begin
  tooth_lengths = fill(2, 3)
  root_vertex = (3, 2)
  c = named_comb_tree(tooth_lengths)
  s = siteinds("S=1/2", c)

  J₁ = 1.0
  J₂ = 0.1

  time_step = 0.1
  time_total = 1.0

  nsites = 2
  maxdim = 100
  cutoff = 1e-8

  s = siteinds("S=1/2", c)
  ℋ₁₀ = ITensorNetworks.heisenberg(c; J1=J₁, J2=0.0)
  ℋ₂₀ = ITensorNetworks.heisenberg(c; J1=0.0, J2=J₂)
  ℋ⃗₀ = [ℋ₁₀, ℋ₂₀]
  H⃗₀ = [TTN(ℋ₀, s) for ℋ₀ in ℋ⃗₀]

  ψ₀ = TTN(ComplexF64, s, v -> iseven(sum(isodd.(v))) ? "↑" : "↓")

  ψₜ_ode = tdvp(ode_updater, H⃗₀, time_total, ψ₀; time_step, maxdim, cutoff, nsites, updater_kwargs=ode_updater_kwargs)

  ψₜ_krylov = tdvp(krylov_updater, H⃗₀, time_total, ψ₀; time_step, cutoff, nsites, updater_kwargs=krylov_updater_kwargs)

  ψₜ_full, _ = tdvp_ode_solver(contract.(H⃗₀), contract(ψ₀); time_step=time_total)

  @test norm(ψ₀) ≈ 1
  @test norm(ψₜ_ode) ≈ 1
  @test norm(ψₜ_krylov) ≈ 1
  @test norm(ψₜ_full) ≈ 1

  ode_err = norm(contract(ψₜ_ode) - ψₜ_full)
  krylov_err = norm(contract(ψₜ_krylov) - ψₜ_full)

  @test krylov_err > ode_err
  @test ode_err < 1e-2
  @test krylov_err < 1e-2
end

nothing
