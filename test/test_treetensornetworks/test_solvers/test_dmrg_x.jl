using ITensors
using ITensorNetworks
using Random
using Test

@testset "MPS DMRG-X" begin
  n = 10
  s = siteinds("S=1/2", n)

  Random.seed!(12)

  W = 12
  # Random fields h ∈ [-W, W]
  h = W * (2 * rand(n) .- 1)
  H = MPO(ITensorNetworks.heisenberg(n; h), s)

  initstate = rand(["↑", "↓"], n)
  ψ = MPS(s, initstate)

  dmrg_x_kwargs = (
    nsweeps=20, reverse_step=false, normalize=true, maxdim=20, cutoff=1e-10, outputlevel=0
  )

  ϕ = dmrg_x(ProjMPO(H), ψ; nsite=2, dmrg_x_kwargs...)

  @test inner(ψ', H, ψ) / inner(ψ, ψ) ≈ inner(ϕ', H, ϕ) / inner(ϕ, ϕ) rtol = 1e-1
  @test inner(H, ψ, H, ψ) ≉ inner(ψ', H, ψ)^2 rtol = 1e-7
  @test inner(H, ϕ, H, ϕ) ≈ inner(ϕ', H, ϕ)^2 rtol = 1e-7

  ϕ̃ = dmrg_x(ProjMPO(H), ϕ; nsite=1, dmrg_x_kwargs...)

  @test inner(ψ', H, ψ) / inner(ψ, ψ) ≈ inner(ϕ̃', H, ϕ̃) / inner(ϕ̃, ϕ̃) rtol = 1e-1
  @test inner(H, ϕ̃, H, ϕ̃) ≈ inner(ϕ̃', H, ϕ̃)^2 rtol = 1e-3
  # Sometimes broken, sometimes not
  # @test abs(loginner(ϕ̃, ϕ) / n) ≈ 0.0 atol = 1e-6
end

@testset "Tree DMRG-X" begin
  tooth_lengths = fill(2, 3)
  root_vertex = (3, 2)
  c = named_comb_tree(tooth_lengths)
  s = siteinds("S=1/2", c)

  Random.seed!(12)

  W = 12
  # Random fields h ∈ [-W, W]
  h = W * (2 * rand(nv(c)) .- 1)
  H = TTN(ITensorNetworks.heisenberg(c; h), s)

  ψ = normalize!(TTN(s, v -> rand(["↑", "↓"])))

  dmrg_x_kwargs = (
    nsweeps=20, reverse_step=false, normalize=true, maxdim=20, cutoff=1e-10, outputlevel=0
  )

  ϕ = dmrg_x(ProjTTN(H), ψ; nsite=2, dmrg_x_kwargs...)

  @test inner(ψ', H, ψ) / inner(ψ, ψ) ≈ inner(ϕ', H, ϕ) / inner(ϕ, ϕ) rtol = 1e-1
  @test inner(H, ψ, H, ψ) ≉ inner(ψ', H, ψ)^2 rtol = 1e-2
  @test inner(H, ϕ, H, ϕ) ≈ inner(ϕ', H, ϕ)^2 rtol = 1e-7

  ϕ̃ = dmrg_x(ProjTTN(H), ϕ; nsite=1, dmrg_x_kwargs...)

  @test inner(ψ', H, ψ) / inner(ψ, ψ) ≈ inner(ϕ̃', H, ϕ̃) / inner(ϕ̃, ϕ̃) rtol = 1e-1
  @test inner(H, ϕ̃, H, ϕ̃) ≈ inner(ϕ̃', H, ϕ̃)^2 rtol = 1e-6
  # Sometimes broken, sometimes not
  # @test abs(loginner(ϕ̃, ϕ) / nv(c)) ≈ 0.0 atol = 1e-8

  # compare against ED
  @disable_warn_order U0 = contract(ψ, root_vertex)
  @disable_warn_order T = contract(H, root_vertex)
  D, U = eigen(T; ishermitian=true)
  u = uniqueind(U, T)
  _, max_ind = findmax(abs, array(dag(U0) * U))
  U_exact = U * dag(onehot(u => max_ind))
  @disable_warn_order U_dmrgx = contract(ϕ, root_vertex)
  @test inner(ϕ', H, ϕ) ≈ (dag(U_exact') * T * U_exact)[] atol = 1e-6
  @test abs(inner(U_dmrgx, U_exact)) ≈ 1 atol = 1e-6
end

nothing
