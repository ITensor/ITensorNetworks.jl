println(
  """

  Test CustomVertexGraphs.jl
  """
)

using Graphs
include(joinpath("..", "src", "CustomVertexGraphs.jl"))
using .CustomVertexGraphs

g = set_vertices(grid((4,)), ["A", "B", "C", "D"])
@show g
@show g[["A", "B"]]
@show has_edge(g, "A" => "B")
@show add_edge!(g, "A" => "C")
@show has_edge(g, "A" => "C")

graph = grid((6,))
verts = [Sub("A")[1], Sub("A")[2], Sub("B")[1], Sub("B")[2], Sub("C")[1], Sub("C")[2]]
g = set_vertices(graph, verts)

@show g
@show g[[Sub("A")[1], Sub("A")[2]]]
@show has_edge(g, Sub("A")[1] => Sub("A")[2])
@show has_edge(g, Sub("A")[2] => Sub("B")[1])
@show !has_edge(g, Sub("A")[2] => Sub("B")[2])
@show add_edge!(g, Sub("A")[1] => Sub("B")[1])
@show has_edge(g, Sub("A")[1] => Sub("B")[1])
@show g[Sub("A")]
@show g[Sub("B")]
@show g[Sub("C")]
@show g[[Sub("A"), Sub("C")]]
@show g[[Sub("A"), Sub("C"), Sub("B")[2]]]

graph = grid((6,))
verts = [Sub("A")["A1"][1], Sub("A")["A1"][2], Sub("A")["A3"][1], Sub("B")["B1"][1], Sub("B")["B2"][1], Sub("B")["B3"][1]]
g = set_vertices(graph, verts)

@show g
@show g[Sub("A")["A1"]]
