@eval module $(gensym())
using Dictionaries: Dictionary
using Graphs: nv, vertices
using ITensorMPS: linkdims
using ITensorNetworks:
  OpSum, ttn, apply, contract, dmrg_x, inner, mpo, mps, random_mps, siteinds
using ITensorNetworks.ModelHamiltonians: ModelHamiltonians
using ITensors: @disable_warn_order, array, dag, onehot, uniqueind
using LinearAlgebra: eigen, normalize
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using StableRNGs: StableRNG
using Test: @test, @testset
# TODO: Combine MPS and TTN tests.
@testset "MPS DMRG-X" for conserve_qns in (false, true)
  n = 10
  s = siteinds("S=1/2", n; conserve_qns)
  W = 12
  # Random fields h ∈ [-W, W]
  rng = StableRNG(1234)
  h = W * (2 * rand(rng, n) .- 1)
  H = mpo(ModelHamiltonians.heisenberg(n; h), s)
  ψ = mps(v -> rand(rng, ["↑", "↓"]), s)
  dmrg_x_kwargs = (nsweeps=20, normalize=true, maxdim=20, cutoff=1e-10, outputlevel=0)
  e, ϕ = dmrg_x(H, ψ; nsites=2, dmrg_x_kwargs...)
  @test inner(ϕ', H, ϕ) / inner(ϕ, ϕ) ≈ e
  @test inner(ψ', H, ψ) / inner(ψ, ψ) ≈ inner(ϕ', H, ϕ) / inner(ϕ, ϕ) rtol = 1e-1
  @test inner(H, ψ, H, ψ) ≉ inner(ψ', H, ψ)^2 rtol = 1e-7
  @test inner(H, ϕ, H, ϕ) ≈ inner(ϕ', H, ϕ)^2 rtol = 1e-7
  e, ϕ̃ = dmrg_x(H, ϕ; nsites=1, dmrg_x_kwargs...)
  @test inner(ϕ̃', H, ϕ̃) / inner(ϕ̃, ϕ̃) ≈ e
  @test inner(ψ', H, ψ) / inner(ψ, ψ) ≈ inner(ϕ̃', H, ϕ̃) / inner(ϕ̃, ϕ̃) rtol = 1e-1
  @test inner(H, ϕ̃, H, ϕ̃) ≈ inner(ϕ̃', H, ϕ̃)^2 rtol = 1e-3
  # Sometimes broken, sometimes not
  # @test abs(loginner(ϕ̃, ϕ) / n) ≈ 0.0 atol = 1e-6
end
@testset "Tree DMRG-X" for conserve_qns in (false, true)
  # TODO: Combine with tests above into a loop over graph structures.
  tooth_lengths = fill(2, 3)
  root_vertex = (3, 2)
  c = named_comb_tree(tooth_lengths)
  s = siteinds("S=1/2", c; conserve_qns)
  W = 12
  # Random fields h ∈ [-W, W]
  rng = StableRNG(123)
  h = Dictionary(vertices(c), W * (2 * rand(rng, nv(c)) .- 1))
  H = ttn(ModelHamiltonians.heisenberg(c; h), s)
  ψ = normalize(ttn(v -> rand(rng, ["↑", "↓"]), s))
  dmrg_x_kwargs = (nsweeps=20, normalize=true, maxdim=20, cutoff=1e-10, outputlevel=0)
  e, ϕ = dmrg_x(H, ψ; nsites=2, dmrg_x_kwargs...)
  @test inner(ϕ', H, ϕ) / inner(ϕ, ϕ) ≈ e
  @test inner(ψ', H, ψ) / inner(ψ, ψ) ≈ inner(ϕ', H, ϕ) / inner(ϕ, ϕ) rtol = 1e-1
  @test inner(H, ψ, H, ψ) ≉ inner(ψ', H, ψ)^2 rtol = 1e-2
  @test inner(H, ϕ, H, ϕ) ≈ inner(ϕ', H, ϕ)^2 rtol = 1e-7
  e, ϕ̃ = dmrg_x(H, ϕ; nsites=1, dmrg_x_kwargs...)
  @test inner(ϕ̃', H, ϕ̃) / inner(ϕ̃, ϕ̃) ≈ e
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
end
