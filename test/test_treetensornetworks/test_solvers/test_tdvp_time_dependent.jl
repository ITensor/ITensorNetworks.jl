@eval module $(gensym())
using ITensors: contract
using ITensorNetworks: ITensorNetworks, TimeDependentSum, ttn, mpo, mps, siteinds, tdvp
using OrdinaryDiffEq: Tsit5
using KrylovKit: exponentiate
using LinearAlgebra: norm
using NamedGraphs: AbstractNamedEdge, named_comb_tree
using Test: @test, @test_broken, @testset

include(
  joinpath(
    @__DIR__, "ITensorNetworksTestSolversUtils", "ITensorNetworksTestSolversUtils.jl"
  ),
)

using .ITensorNetworksTestSolversUtils:
  ITensorNetworksTestSolversUtils, krylov_solver, ode_solver

# Functions need to be defined in global scope (outside
# of the @testset macro)

ω₁ = 0.1
ω₂ = 0.2

ode_alg = Tsit5()
ode_kwargs = (; reltol=1e-8, abstol=1e-8)

ω⃗ = [ω₁, ω₂]
f⃗ = [t -> cos(ω * t) for ω in ω⃗]
ode_updater_kwargs = (; f=[f⃗], solver_alg=ode_alg, ode_kwargs)

function ode_updater(
  init;
  state!,
  projected_operator!,
  outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  internal_kwargs,
  ode_kwargs,
  solver_alg,
  f,
)
  region = first(sweep_plan[which_region_update])
  (; time_step, t) = internal_kwargs
  t = isa(region, AbstractNamedEdge) ? t : t + time_step

  H⃗₀ = projected_operator![]
  result, info = ode_solver(
    -im * TimeDependentSum(f, H⃗₀),
    time_step,
    init;
    current_time=t,
    solver_alg,
    ode_kwargs...,
  )
  return result, (; info)
end

function tdvp_ode_solver(H⃗₀, ψ₀; time_step, kwargs...)
  psi_t, info = ode_solver(
    -im * TimeDependentSum(f⃗, H⃗₀), time_step, ψ₀; solver_alg=ode_alg, ode_kwargs...
  )
  return psi_t, (; info)
end

krylov_kwargs = (; tol=1e-8, krylovdim=15, eager=true)
krylov_updater_kwargs = (; f=[f⃗], krylov_kwargs)

function ITensorNetworksTestSolversUtils.krylov_solver(
  H⃗₀, ψ₀; time_step, ishermitian=false, issymmetric=false, kwargs...
)
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
  internal_kwargs,
  ishermitian=false,
  issymmetric=false,
  f,
  krylov_kwargs,
)
  (; time_step, t) = internal_kwargs
  H⃗₀ = projected_operator![]
  region = first(sweep_plan[which_region_update])
  t = isa(region, AbstractNamedEdge) ? t : t + time_step

  result, info = krylov_solver(
    -im * TimeDependentSum(f, H⃗₀),
    time_step,
    init;
    current_time=t,
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

  ψₜ_ode = tdvp(
    H⃗₀,
    time_total,
    ψ₀;
    time_step,
    maxdim,
    cutoff,
    nsites,
    updater=ode_updater,
    updater_kwargs=ode_updater_kwargs,
  )

  ψₜ_krylov = tdvp(
    H⃗₀,
    time_total,
    ψ₀;
    time_step,
    cutoff,
    nsites,
    updater=krylov_updater,
    updater_kwargs=krylov_updater_kwargs,
  )

  ψₜ_full, _ = tdvp_ode_solver(contract.(H⃗₀), contract(ψ₀); time_step=time_total)

  @test norm(ψ₀) ≈ 1
  @test norm(ψₜ_ode) ≈ 1
  @test norm(ψₜ_krylov) ≈ 1
  @test norm(ψₜ_full) ≈ 1

  ode_err = norm(contract(ψₜ_ode) - ψₜ_full)
  krylov_err = norm(contract(ψₜ_krylov) - ψₜ_full)
  #ToDo: Investigate why Krylov gives better result than ODE solver
  @test_broken krylov_err > ode_err
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

  ψ₀ = ttn(ComplexF64, s, v -> iseven(sum(isodd.(v))) ? "↑" : "↓")

  ψₜ_ode = tdvp(
    H⃗₀,
    time_total,
    ψ₀;
    time_step,
    maxdim,
    cutoff,
    nsites,
    updater=ode_updater,
    updater_kwargs=ode_updater_kwargs,
  )

  ψₜ_krylov = tdvp(
    H⃗₀,
    time_total,
    ψ₀;
    time_step,
    cutoff,
    nsites,
    updater=krylov_updater,
    updater_kwargs=krylov_updater_kwargs,
  )
  ψₜ_full, _ = tdvp_ode_solver(contract.(H⃗₀), contract(ψ₀); time_step=time_total)

  @test norm(ψ₀) ≈ 1
  @test norm(ψₜ_ode) ≈ 1
  @test norm(ψₜ_krylov) ≈ 1
  @test norm(ψₜ_full) ≈ 1

  ode_err = norm(contract(ψₜ_ode) - ψₜ_full)
  krylov_err = norm(contract(ψₜ_krylov) - ψₜ_full)
  #ToDo: Investigate why Krylov gives better result than ODE solver
  @test_broken krylov_err > ode_err
  @test ode_err < 1e-2
  @test krylov_err < 1e-2
end
end
