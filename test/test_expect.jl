@eval module $(gensym())
using Test
using ITensorNetworks

using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph, vertices
using NamedGraphs.NamedGraphGenerators: named_grid
using ITensors: siteinds
using ITensorNetworks:
  BeliefPropagationCache, expect, random_tensornetwork, original_state_vertex
using Random: Random
using SplitApplyCombine: group

@testset "Test Expect" begin
  Random.seed!(1234)

  #Test on a tree
  L, χ = 4, 2
  g = NamedGraph(SimpleGraph(uniform_tree(L)))
  s = siteinds("S=1/2", g)
  ψ = random_tensornetwork(s; link_space=χ)
  sz_bp = expect(ψ, "Sz"; alg="bp")
  sz_exact = expect(ψ, "Sz"; alg="exact")
  @test sz_bp ≈ sz_exact

  #Test on a grid, group by column to make BP exact
  L, χ = 2, 2
  g = named_grid((L, L))
  s = siteinds("S=1/2", g)
  ψ = random_tensornetwork(s; link_space=χ)
  cache_construction_function =
    f -> BeliefPropagationCache(
      f; partitioned_vertices=group(v -> (original_state_vertex(f, v)[1]), vertices(f))
    )
  sz_bp = expect(ψ, "Sz"; alg="bp", cache_construction_function)
  sz_exact = expect(ψ, "Sz"; alg="exact")
  @test sz_bp ≈ sz_exact

  #Test with QNS, product state so should be immediately exact
  #TODO: Fix Construct TN with QNS that's got a bond dimension bigger than 1. Add() is broken...
  L, χ = 2, 2
  g = named_grid((L, L))
  s = siteinds("S=1/2", g; conserve_qns=true)
  ψ = ITensorNetwork(v -> isodd(sum(v)) ? "↑" : "↓", s)

  sz_bp = expect(ψ, "Sz"; alg="bp")
  sz_exact = expect(ψ, "Sz"; alg="exact")
  @test sz_bp ≈ sz_exact
end
end
