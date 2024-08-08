@eval module $(gensym())
using Graphs: rem_edge!, vertices
using NamedGraphs: NamedEdge
using NamedGraphs.NamedGraphGenerators: named_grid
using ITensorNetworks: ITensorNetwork, inner_network, random_tensornetwork, siteinds
using ITensors: ITensors, apply, op, scalar, inner
using LinearAlgebra: norm_sqr
using StableRNGs: StableRNG
using Test: @test, @testset
@testset "add_itensornetworks" begin
  g = named_grid((2, 2))
  s = siteinds("S=1/2", g)
  ψ1 = ITensorNetwork(v -> "↑", s)
  ψ2 = ITensorNetwork(v -> "↓", s)

  ψ_GHZ = ψ1 + ψ2

  v = (2, 2)
  Oψ_GHZ = copy(ψ_GHZ)
  Oψ_GHZ[v] = apply(op("Sz", s[v]), Oψ_GHZ[v])

  ψψ_GHZ = inner_network(ψ_GHZ, ψ_GHZ)
  ψOψ_GHZ = inner_network(ψ_GHZ, Oψ_GHZ)

  @test scalar(ψOψ_GHZ) / scalar(ψψ_GHZ) == 0.0

  χ = 3
  s1 = siteinds("S=1/2", g)
  s2 = copy(s1)
  rem_edge!(s2, NamedEdge((1, 1) => (1, 2)))

  v = rand(vertices(g))
  rng = StableRNG(1234)
  ψ1 = random_tensornetwork(rng, s1; link_space=χ)
  ψ2 = random_tensornetwork(rng, s2; link_space=χ)

  ψ12 = ψ1 + ψ2

  Oψ12 = copy(ψ12)
  Oψ12[v] = apply(op("Sz", s1[v]), Oψ12[v])

  Oψ1 = copy(ψ1)
  Oψ1[v] = apply(op("Sz", s1[v]), Oψ1[v])

  Oψ2 = copy(ψ2)
  Oψ2[v] = apply(op("Sz", s2[v]), Oψ2[v])

  alg = "exact"
  expec_method1 =
    (inner(ψ1, Oψ1; alg) + inner(ψ2, Oψ2; alg) + 2 * inner(ψ1, Oψ2; alg)) /
    (norm_sqr(ψ1; alg) + norm_sqr(ψ2; alg) + 2 * inner(ψ1, ψ2; alg))
  expec_method2 = inner(ψ12, Oψ12; alg) / norm_sqr(ψ12; alg)

  @test expec_method1 ≈ expec_method2
end
end
