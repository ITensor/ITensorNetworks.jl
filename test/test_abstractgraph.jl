@eval module $(gensym())
using Graphs: add_edge!, add_vertex!
using NamedGraphs.GraphsExtensions: is_binary_arborescence, is_rooted, root_vertex
using NamedGraphs: NamedDiGraph
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
