@eval module $(gensym())
using Graphs: SimpleGraph, uniform_tree
using NamedGraphs: NamedGraph, vertices
using NamedGraphs.NamedGraphGenerators: named_grid
using ITensors: siteinds
using ITensorNetworks:
  BeliefPropagationCache,
  ITensorNetwork,
  expect,
  random_tensornetwork,
  original_state_vertex
using SplitApplyCombine: group
using StableRNGs: StableRNG
using Test: @test, @testset
@testset "Test Expect" begin
  #Test on a tree
  L, χ = 4, 2
  g = NamedGraph(SimpleGraph(uniform_tree(L)))
  s = siteinds("S=1/2", g)
  rng = StableRNG(1234)
  ψ = random_tensornetwork(rng, s; link_space=χ)
  sz_bp = expect(ψ, "Sz"; alg="bp")
  sz_exact = expect(ψ, "Sz"; alg="exact")
  @test sz_bp ≈ sz_exact

  #Test on a grid, group by column to make BP exact
  L, χ = 2, 2
  g = named_grid((L, L))
  s = siteinds("S=1/2", g)
  rng = StableRNG(1234)
  ψ = random_tensornetwork(rng, s; link_space=χ)
  quadratic_form_vertices = reduce(
    vcat, [[(v, "ket"), (v, "bra"), (v, "operator")] for v in vertices(ψ)]
  )
  cache_construction_kwargs = (;
    partitioned_vertices=group(v -> first(first(v)), quadratic_form_vertices)
  )
  sz_bp = expect(ψ, "Sz"; alg="bp", cache_construction_kwargs)
  sz_exact = expect(ψ, "Sz"; alg="exact")
  @test sz_bp ≈ sz_exact

  #Test with QNS, product state so should be immediately exact
  L, χ = 2, 2
  g = named_grid((L, L))
  s = siteinds("S=1/2", g; conserve_qns=true)
  ψ = ITensorNetwork(v -> isodd(sum(v)) ? "↑" : "↓", s)

  sz_bp = expect(ψ, "Sz"; alg="bp")
  sz_exact = expect(ψ, "Sz"; alg="exact")
  @test sz_bp ≈ sz_exact
end
end
