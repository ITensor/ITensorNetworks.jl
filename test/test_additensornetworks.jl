using ITensorNetworks
using ITensorNetworks: add_itensornetworks, inner_network
using Test
using Compat
using ITensors
using Metis
using NamedGraphs
using NamedGraphs: hexagonal_lattice_graph
using Random
using LinearAlgebra
using SplitApplyCombine

using Random

@testset "add_itensornetworks" begin
  Random.seed!(5623)
  g = named_grid((2, 3))
  s = siteinds("S=1/2", g)
  ψ1 = ITensorNetwork(s, v -> "↑")
  ψ2 = ITensorNetwork(s, v -> "↓")

  ψ_GHZ = add_itensornetworks(ψ1, ψ2)

  v = (2, 2)
  Oψ_GHZ = copy(ψ_GHZ)
  Oψ_GHZ[v] = apply(op("Sz", s[v]), Oψ_GHZ[v])

  ψψ_GHZ = inner_network(ψ_GHZ, ψ_GHZ)
  ψOψ_GHZ = inner_network(ψ_GHZ, Oψ_GHZ)

  @test ITensors.contract(ψOψ_GHZ)[] / ITensors.contract(ψψ_GHZ)[] == 0.0

  χ = 3
  g = hexagonal_lattice_graph(1, 2)
  s = siteinds("S=1/2", g)

  v = rand(vertices(g))
  ψ1 = randomITensorNetwork(s; link_space=χ)
  ψ2 = randomITensorNetwork(s; link_space=χ)

  ψ12 = add_itensornetworks(ψ1, ψ2)

  Oψ12 = copy(ψ12)
  Oψ12[v] = apply(op("Sz", s[v]), Oψ12[v])

  Oψ1 = copy(ψ1)
  Oψ1[v] = apply(op("Sz", s[v]), Oψ1[v])

  Oψ2 = copy(ψ2)
  Oψ2[v] = apply(op("Sz", s[v]), Oψ2[v])

  ψψ_12 = inner_network(ψ12, ψ12)
  ψOψ_12 = inner_network(ψ12, Oψ12)

  ψ1ψ2 = inner_network(ψ1, ψ2)
  ψ1Oψ2 = inner_network(ψ1, Oψ2)

  ψψ_2 = inner_network(ψ2, ψ2)
  ψOψ_2 = inner_network(ψ2, Oψ2)

  ψψ_1 = inner_network(ψ1, ψ1)
  ψOψ_1 = inner_network(ψ1, Oψ1)

  e12_alt =
    (
      ITensors.contract(ψOψ_1)[] +
      ITensors.contract(ψOψ_2)[] +
      2 * ITensors.contract(ψ1Oψ2)[]
    ) /
    (ITensors.contract(ψψ_1)[] + ITensors.contract(ψψ_2)[] + 2 * ITensors.contract(ψ1ψ2)[])
  e12 = ITensors.contract(ψOψ_12)[] / ITensors.contract(ψψ_12)[]

  @test e12 ≈ e12_alt
end
