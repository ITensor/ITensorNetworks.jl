@eval module $(gensym())
using Graphs: add_edge!, add_vertex!
using NamedGraphs: NamedDiGraph
using NamedGraphs.GraphsExtensions: root_vertex, is_rooted, is_binary_arborescence
using Test: @test, @testset

@testset "test rooted directed graphs" begin
  g = NamedDiGraph([1, 2, 3])
  @test !is_rooted(g)
  add_edge!(g, 1, 2)
  add_edge!(g, 1, 3)
  @test is_rooted(g)
  @test root_vertex(g) == 1
  @test is_binary_arborescence(g)
  add_vertex!(g, 4)
  add_edge!(g, 1, 4)
  @test !is_binary_arborescence(g)
end
end
