@eval module $(gensym())
using Graphs: rem_edge!, vertices
using NamedGraphs: NamedEdge, hexagonal_lattice_graph, named_grid
using ITensorNetworks: ITensorNetwork, inner_network, random_tensornetwork, siteinds
using ITensors: ITensors, apply, contract, op
using Random: Random
using Test: @test, @testset

@testset "add_itensornetworks" begin
  Random.seed!(5623)
  g = named_grid((2, 3))
  s = siteinds("S=1/2", g)
  ψ1 = ITensorNetwork(v -> "↑", s)
  ψ2 = ITensorNetwork(v -> "↓", s)

  ψ_GHZ = ψ1 + ψ2

  v = (2, 2)
  Oψ_GHZ = copy(ψ_GHZ)
  Oψ_GHZ[v] = apply(op("Sz", s[v]), Oψ_GHZ[v])

  ψψ_GHZ = inner_network(ψ_GHZ, ψ_GHZ)
  ψOψ_GHZ = inner_network(ψ_GHZ, Oψ_GHZ)

  @test contract(ψOψ_GHZ)[] / contract(ψψ_GHZ)[] == 0.0

  χ = 3
  g = hexagonal_lattice_graph(1, 2)

  s1 = siteinds("S=1/2", g)
  s2 = copy(s1)
  rem_edge!(s2, NamedEdge((1, 1) => (1, 2)))

  v = rand(vertices(g))
  ψ1 = random_tensornetwork(s1; link_space=χ)
  ψ2 = random_tensornetwork(s2; link_space=χ)

  ψ12 = ψ1 + ψ2

  Oψ12 = copy(ψ12)
  Oψ12[v] = apply(op("Sz", s1[v]), Oψ12[v])

  Oψ1 = copy(ψ1)
  Oψ1[v] = apply(op("Sz", s1[v]), Oψ1[v])

  Oψ2 = copy(ψ2)
  Oψ2[v] = apply(op("Sz", s2[v]), Oψ2[v])

  ψψ_12 = inner_network(ψ12, ψ12)
  ψOψ_12 = inner_network(ψ12, Oψ12)

  ψ1ψ2 = inner_network(ψ1, ψ2)
  ψ1Oψ2 = inner_network(ψ1, Oψ2)

  ψψ_2 = inner_network(ψ2, ψ2)
  ψOψ_2 = inner_network(ψ2, Oψ2)

  ψψ_1 = inner_network(ψ1, ψ1)
  ψOψ_1 = inner_network(ψ1, Oψ1)

  expec_method1 =
    (contract(ψOψ_1)[] + contract(ψOψ_2)[] + 2 * contract(ψ1Oψ2)[]) /
    (contract(ψψ_1)[] + contract(ψψ_2)[] + 2 * contract(ψ1ψ2)[])
  expec_method2 = contract(ψOψ_12)[] / contract(ψψ_12)[]

  @test expec_method1 ≈ expec_method2
end
end
