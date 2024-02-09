using ITensors
using Graphs
using NamedGraphs
using ITensorNetworks
using ITensorNetworks:
  delta_network,
  gradient,
  update,
  tensornetwork,
  bra_vertex_map,
  ket_vertex_map,
  dual_index_map
using Test
using Random

@testset "FormNetworkss" begin
  g = named_grid((1, 4))
  s_ket = siteinds("S=1/2", g)
  s_bra = prime(s_ket; links=[])
  s_operator = union_all_inds(s_bra, s_ket)
  χ = 2
  Random.seed!(1234)
  ψket = randomITensorNetwork(s_ket; link_space=χ)
  ψbra = randomITensorNetwork(s_bra; link_space=χ)
  A = delta_network(s_operator)

  blf = BilinearFormNetwork(A, ψbra, ψket)
  @test nv(blf) == nv(ψket) + nv(ψbra) + nv(A)
  @test isempty(externalinds(blf))

  qf = QuadraticFormNetwork(A, ψket)
  @test nv(qf) == 2 * nv(ψbra) + nv(A)
  @test isempty(externalinds(qf))

  v = (1, 1)
  new_tensor = randomITensor(inds(ψket[v]))
  qf_updated = update(qf, v, copy(new_tensor))

  @test tensornetwork(qf_updated)[bra_vertex_map(qf_updated)(v)] ≈
    dual_index_map(qf_updated)(dag(new_tensor))
  @test tensornetwork(qf_updated)[ket_vertex_map(qf_updated)(v)] ≈ new_tensor
end
