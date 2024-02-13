using ITensors
using Graphs
using NamedGraphs
using ITensorNetworks
using ITensorNetworks:
  delta_network,
  update,
  tensornetwork,
  bra_vertex_map,
  ket_vertex_map,
  dual_index_map,
  bra_network,
  ket_network,
  operator_network
using Test
using Random

@testset "FormNetworkss" begin
  g = named_grid((1, 4))
  s_ket = siteinds("S=1/2", g)
  s_bra = prime(s_ket; links=[])
  s_operator = union_all_inds(s_bra, s_ket)
  χ, D = 2, 3
  Random.seed!(1234)
  ψket = randomITensorNetwork(s_ket; link_space=χ)
  ψbra = randomITensorNetwork(s_bra; link_space=χ)
  A = randomITensorNetwork(s_operator; link_space=D)

  blf = BilinearFormNetwork(A, ψbra, ψket)
  @test nv(blf) == nv(ψket) + nv(ψbra) + nv(A)
  @test isempty(externalinds(blf))

  @test underlying_graph(ket_network(blf)) == underlying_graph(ψket)
  @test underlying_graph(operator_network(blf)) == underlying_graph(A)
  @test underlying_graph(bra_network(blf)) == underlying_graph(ψbra)

  qf = QuadraticFormNetwork(A, ψket)
  @test nv(qf) == 2 * nv(ψbra) + nv(A)
  @test isempty(externalinds(qf))

  v = (1, 1)
  new_tensor = randomITensor(inds(ψket[v]))
  qf_updated = update(qf, v, copy(new_tensor))

  @test tensornetwork(qf_updated)[bra_vertex_map(qf_updated)(v)] ≈
    dual_index_map(qf_updated)(dag(new_tensor))
  @test tensornetwork(qf_updated)[ket_vertex_map(qf_updated)(v)] ≈ new_tensor

  @test underlying_graph(ket_network(qf)) == underlying_graph(ψket)
  @test underlying_graph(operator_network(qf)) == underlying_graph(A)
end
