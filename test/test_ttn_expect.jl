@eval module $(gensym())
using Graphs: vertices
using ITensorNetworks: ttn, expect, random_mps, siteinds
using LinearAlgebra: norm
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using StableRNGs: StableRNG
using Test: @test, @testset
@testset "TTN expect" begin
  tooth_lengths = fill(2, 2)
  c = named_comb_tree(tooth_lengths)
  s = siteinds("S=1/2", c)
  d = Dict()
  magnetization = Dict()
  for (i, v) in enumerate(vertices(s))
    d[v] = isodd(i) ? "Up" : "Dn"
    magnetization[v] = isodd(i) ? 0.5 : -0.5
  end
  states = v -> d[v]
  state = ttn(states, s)
  res = expect("Sz", state)
  @test all([isapprox(res[v], magnetization[v]; atol=1e-8) for v in vertices(s)])
end
end
