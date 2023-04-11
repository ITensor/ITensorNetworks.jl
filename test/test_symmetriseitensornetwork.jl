using ITensors
using ITensorNetworks
using ITensorNetworks:
  compute_message_tensors, nested_graph_leaf_vertices, contract_inner, symmetric_gauge
using NamedGraphs
using Test
using Compat
using Random
using SplitApplyCombine

@testset "symmetrise_itensornetwork" begin
  n = 3
  dims = (n, n)
  g = named_grid(dims)
  s = siteinds("S=1/2", g)
  χ = 3

  Random.seed!(5467)
  ψ = randomITensorNetwork(s; link_space=χ)
  ψ_symm, ψ_symm_mts = symmetric_gauge(ψ; niters=50)

  #Test we just did a gauge transform and didn't change the overall network
  @test contract_inner(ψ_symm, ψ) /
        sqrt(contract_inner(ψ_symm, ψ_symm) * contract_inner(ψ, ψ)) ≈ 1.0

  ψψ_symm = ψ_symm ⊗ prime(dag(ψ_symm); sites=[])
  vertex_groups = nested_graph_leaf_vertices(
    partition(ψψ_symm, group(v -> v[1], vertices(ψψ_symm)))
  )
  ψ_symm_mts_V2 = compute_message_tensors(ψψ_symm; vertex_groups=vertex_groups, niters=50)

  for e in edges(ψ_symm_mts_V2)
    #Test all message tensors are approximately diagonal
    m_e = ψ_symm_mts_V2[e][first(vertices(ψ_symm_mts_V2[e]))]
    @test diagITensor(vector(diag(m_e)), inds(m_e)) ≈ m_e atol = 1e-8
  end
end
