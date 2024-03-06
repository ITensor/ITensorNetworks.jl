using ITensors
using ITensorNetworks
using ITensorNetworks:
  contract_inner,
  symmetric_gauge,
  gauge_error,
  update,
  messages,
  BeliefPropagationCache,
  VidalITensorNetwork
using NamedGraphs
using Test
using Compat
using Random
using SplitApplyCombine

@testset "gauging" begin
  n = 3
  dims = (n, n)
  g = named_grid(dims)
  s = siteinds("S=1/2", g)
  χ = 6

  Random.seed!(5467)
  ψ = randomITensorNetwork(s; link_space=χ)

  # Move to symmetric gauge
  ψ_symm, bp_cache = symmetric_gauge(ψ; cache_update_kwargs=(; maxiter=20))

  # Test we just did a gauge transform and didn't change the overall network
  @test contract_inner(ψ_symm, ψ) /
        sqrt(contract_inner(ψ_symm, ψ_symm) * contract_inner(ψ, ψ)) ≈ 1.0

  #Test all message tensors are approximately diagonal even when we keep running BP
  bp_cache = update(bp_cache; maxiter=20)
  for m_e in values(messages(bp_cache))
    @test diagITensor(vector(diag(only(m_e))), inds(only(m_e))) ≈ only(m_e) atol = 1e-8
  end

  # Move directly to vidal gauge
  ψ_vidal = VidalITensorNetwork(ψ)
  @test gauge_error(ψ_vidal) < 1e-5

  #Move from vidal to symmetric gauge
  ψ_symm, bp_cache = symmetric_gauge(ψ_vidal)
  for m_e in values(messages(bp_cache))
    @test diagITensor(vector(diag(only(m_e))), inds(only(m_e))) ≈ only(m_e) atol = 1e-8
  end
end
