using Test
using ITensorNetworks
using ITensorNetworks: NamedGraph, NamedDimGraph, NamedDimDiGraph, parent_graph
using Random
using Graphs: Graph, DiGraph, grid, add_edge!, rem_edge!

# TODO: remove once this is merged into NamedGraphs.jl

@testset "AbstractNamedGraph equality" begin
  # NamedGraph
  g = grid((2, 2))
  vs = ["A", "B", "C", "D"]
  ng1 = NamedGraph(g, vs)
  # construct same NamedGraph with different underlying structure
  ng2 = NamedGraph(Graph(4), vs[[1, 4, 3, 2]])
  add_edge!(ng2, "A" => "B")
  add_edge!(ng2, "A" => "C")
  add_edge!(ng2, "B" => "D")
  add_edge!(ng2, "C" => "D")
  @test parent_graph(ng1) != parent_graph(ng2)
  @test ng1 == ng2
  rem_edge!(ng2, "B" => "A")
  @test ng1 != ng2

  # NamedDimGraph
  dvs = [("X", 1), ("X", 2), ("Y", 1), ("Y", 2)]
  ndg1 = NamedDimGraph(g, dvs)
  # construct same NamedDimGraph from different underlying structure
  ndg2 = NamedDimGraph(Graph(4), dvs[[1, 4, 3, 2]])
  add_edge!(ndg2, ("X", 1) => ("X", 2))
  add_edge!(ndg2, ("X", 1) => ("Y", 1))
  add_edge!(ndg2, ("X", 2) => ("Y", 2))
  add_edge!(ndg2, ("Y", 1) => ("Y", 2))
  @test parent_graph(ndg1) != parent_graph(ndg2)
  @test ndg1 == ndg2
  rem_edge!(ndg2, ("Y", 1) => ("X", 1))
  @test ndg1 != ndg2

  # NamedDimDiGraph
  nddg1 = NamedDimDiGraph(DiGraph(collect(edges(g))), dvs)
  # construct same NamedDimDiGraph from different underlying structure
  nddg2 = NamedDimDiGraph(DiGraph(4), dvs[[1, 4, 3, 2]])
  add_edge!(nddg2, ("X", 1) => ("X", 2))
  add_edge!(nddg2, ("X", 1) => ("Y", 1))
  add_edge!(nddg2, ("X", 2) => ("Y", 2))
  add_edge!(nddg2, ("Y", 1) => ("Y", 2))
  @test parent_graph(nddg1) != parent_graph(nddg2)
  @test nddg1 == nddg2
  rem_edge!(nddg2, ("X", 1) => ("Y", 1))
  add_edge!(nddg2, ("Y", 1) => ("X", 1))
  @test nddg1 != nddg2
end

@testset "AbstractNamedGraph vertex renaming" begin
  g = grid((2, 2))
  integer_names = collect(1:4)
  string_names = ["A", "B", "C", "D"]
  tuple_names = [("X", 1), ("X", 2), ("Y", 1), ("Y", 2)]
  function_name = x -> reverse(x)

  # NamedGraph
  ng = NamedGraph(g, string_names)
  # rename to integers
  vmap_int = Dictionary(vertices(ng), integer_names)
  ng_int = rename_vertices(ng, vmap_int)
  @test isa(ng_int, NamedGraph{Int})
  @test has_vertex(ng_int, 3)
  @test has_edge(ng_int, 1 => 2)
  @test has_edge(ng_int, 2 => 4)
  # rename to tuples
  vmap_tuple = Dictionary(vertices(ng), tuple_names)
  ng_tuple = rename_vertices(ng, vmap_tuple)
  @test isa(ng_tuple, NamedGraph{Tuple{String,Int}})
  @test has_vertex(ng_tuple, ("X", 1))
  @test has_edge(ng_tuple, ("X", 1) => ("X", 2))
  @test has_edge(ng_tuple, ("X", 2) => ("Y", 2))
  # rename with name map function
  ng_function = rename_vertices(ng_tuple, function_name)
  @test isa(ng_function, NamedGraph{Tuple{Int,String}})
  @test has_vertex(ng_function, (1, "X"))
  @test has_edge(ng_function, (1, "X") => (2, "X"))
  @test has_edge(ng_function, (2, "X") => (2, "Y"))

  # NamedDimGraph
  ndg = named_grid((2, 2))
  # rename to integers
  vmap_int = Dictionary(vertices(ndg), integer_names)
  ndg_int = rename_vertices(ndg, vmap_int)
  @test isa(ndg_int, NamedDimGraph{Tuple})
  @test has_vertex(ndg_int, (1,))
  @test has_edge(ndg_int, (1,) => (2,))
  @test has_edge(ndg_int, (2,) => (4,))
  # rename to strings
  vmap_string = Dictionary(vertices(ndg), string_names)
  ndg_string = rename_vertices(ndg, vmap_string)
  @test isa(ndg_string, NamedDimGraph{Tuple})
  @test has_vertex(ndg_string, ("A",))
  @test has_edge(ndg_string, ("A",) => ("B",))
  @test has_edge(ndg_string, ("B",) => ("D",))
  # rename to strings
  vmap_tuple = Dictionary(vertices(ndg), tuple_names)
  ndg_tuple = rename_vertices(ndg, vmap_tuple)
  @test isa(ndg_tuple, NamedDimGraph{Tuple})
  @test has_vertex(ndg_tuple, "X", 1)
  @test has_edge(ndg_tuple, ("X", 1) => ("X", 2))
  @test has_edge(ndg_tuple, ("X", 2) => ("Y", 2))
  # rename with name map function
  ndg_function = rename_vertices(ndg_tuple, function_name)
  @test isa(ndg_function, NamedDimGraph{Tuple})
  @test has_vertex(ndg_function, 1, "X")
  @test has_edge(ndg_function, (1, "X") => (2, "X"))
  @test has_edge(ndg_function, (2, "X") => (2, "Y"))

  # NamedDimDiGraph
  nddg = NamedDimDiGraph(DiGraph(collect(edges(g))), vertices(ndg))
  # rename to integers
  vmap_int = Dictionary(vertices(nddg), integer_names)
  nddg_int = rename_vertices(nddg, vmap_int)
  @test isa(nddg_int, NamedDimDiGraph{Tuple})
  @test has_vertex(nddg_int, (1,))
  @test has_edge(nddg_int, (1,) => (2,))
  @test has_edge(nddg_int, (2,) => (4,))
  # rename to strings
  vmap_string = Dictionary(vertices(nddg), string_names)
  nddg_string = rename_vertices(nddg, vmap_string)
  @test isa(nddg_string, NamedDimDiGraph{Tuple})
  @test has_vertex(nddg_string, ("A",))
  @test has_edge(nddg_string, ("A",) => ("B",))
  @test has_edge(nddg_string, ("B",) => ("D",))
  @test !has_edge(nddg_string, ("D",) => ("B",))
  # rename to strings
  vmap_tuple = Dictionary(vertices(nddg), tuple_names)
  nddg_tuple = rename_vertices(nddg, vmap_tuple)
  @test isa(nddg_tuple, NamedDimDiGraph{Tuple})
  @test has_vertex(nddg_tuple, "X", 1)
  @test has_edge(nddg_tuple, ("X", 1) => ("X", 2))
  @test !has_edge(nddg_tuple, ("Y", 2) => ("X", 2))
  # rename with name map function
  nddg_function = rename_vertices(nddg_tuple, function_name)
  @test isa(nddg_function, NamedDimDiGraph{Tuple})
  @test has_vertex(nddg_function, 1, "X")
  @test has_edge(nddg_function, (1, "X") => (2, "X"))
  @test has_edge(nddg_function, (2, "X") => (2, "Y"))
end
