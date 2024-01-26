using ITensors
using ITensorNetworks
using ITensorNetworks:
  belief_propagation,
  contract_inner,
  symmetric_gauge,
  symmetric_to_vidal_gauge,
  vidal_itn_canonicalness,
  vidal_gauge,
  symmetric_itn_canonicalness
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
  ψ_symm, pψψ_symm, ψ_symm_mts = symmetric_gauge(ψ; niters=50)

  @test symmetric_itn_canonicalness(ψ_symm, pψψ_symm, ψ_symm_mts) < 1e-5

  #Test we just did a gauge transform and didn't change the overall network
  @test contract_inner(ψ_symm, ψ) /
        sqrt(contract_inner(ψ_symm, ψ_symm) * contract_inner(ψ, ψ)) ≈ 1.0

  ψψ_symm_V2 = ψ_symm ⊗ prime(dag(ψ_symm); sites=[])
  pψψ_symm_V2 = PartitionedGraph(ψψ_symm_V2, group(v -> v[1], vertices(ψψ_symm_V2)))
  ψ_symm_mts_V2 = belief_propagation(pψψ_symm_V2; niters=50)

  for m_e in values(ψ_symm_mts_V2)
    #Test all message tensors are approximately diagonal
    @test diagITensor(vector(diag(only(m_e))), inds(only(m_e))) ≈ only(m_e) atol = 1e-8
  end

  ψ_vidal, bond_tensors = vidal_gauge(ψ; target_canonicalness=1e-6)
  @test vidal_itn_canonicalness(ψ_vidal, bond_tensors) < 1e-5

  ψ_vidal, bond_tensors = symmetric_to_vidal_gauge(ψ_symm, pψψ_symm, ψ_symm_mts)
  @test vidal_itn_canonicalness(ψ_vidal, bond_tensors) < 1e-5
end
