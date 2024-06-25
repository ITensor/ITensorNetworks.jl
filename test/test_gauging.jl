@eval module $(gensym())
using Compat: Compat
using ITensorNetworks:
  BeliefPropagationCache,
  ITensorNetwork,
  VidalITensorNetwork,
  gauge_error,
  messages,
  random_tensornetwork,
  siteinds,
  update
using ITensors: diag_itensor, inds, inner
using ITensors.NDTensors: vector
using LinearAlgebra: diag
using NamedGraphs.NamedGraphGenerators: named_grid
using StableRNGs: StableRNG
using Test: @test, @testset

@testset "gauging" begin
  n = 3
  dims = (n, n)
  g = named_grid(dims)
  s = siteinds("S=1/2", g)
  χ = 6

  rng = StableRNG(1234)
  ψ = random_tensornetwork(rng, s; link_space=χ)

  # Move directly to vidal gauge
  ψ_vidal = VidalITensorNetwork(ψ; cache_update_kwargs=(; maxiter=30, verbose=true))
  @test gauge_error(ψ_vidal) < 1e-8

  # Move to symmetric gauge
  cache_ref = Ref{BeliefPropagationCache}()
  ψ_symm = ITensorNetwork(ψ_vidal; (cache!)=cache_ref)
  bp_cache = cache_ref[]

  # Test we just did a gauge transform and didn't change the overall network
  @test inner(ψ_symm, ψ) / sqrt(inner(ψ_symm, ψ_symm) * inner(ψ, ψ)) ≈ 1.0 atol = 1e-8

  #Test all message tensors are approximately diagonal even when we keep running BP
  bp_cache = update(bp_cache; maxiter=10)
  for m_e in values(messages(bp_cache))
    @test diag_itensor(vector(diag(only(m_e))), inds(only(m_e))) ≈ only(m_e) atol = 1e-8
  end
end
end
