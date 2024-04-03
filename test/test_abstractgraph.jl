@eval module $(gensym())
using NamedGraphs: add_edge!, add_vertex!, NamedDiGraph
using ITensorNetworks: _root, _is_rooted, _is_rooted_directed_binary_tree
using Test: @test, @testset

@testset "test rooted directed graphs" begin
  g = NamedDiGraph([1, 2, 3])
  @test !_is_rooted(g)
  add_edge!(g, 1, 2)
  add_edge!(g, 1, 3)
  @test _is_rooted(g)
  @test _root(g) == 1
  @test _is_rooted_directed_binary_tree(g)
  add_vertex!(g, 4)
  add_edge!(g, 1, 4)
  @test !_is_rooted_directed_binary_tree(g)
end
end
