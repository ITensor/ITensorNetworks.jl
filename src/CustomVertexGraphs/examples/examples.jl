println(
  """

  Test AbstractBijections
  """
)

include("AbstractBijections.jl")

@show f = Bijection(["X1", "X2"], ["Y1", "Y2"])
@show f["X1"]
@show inv(f)["Y1"]
@show domain(f)
@show image(f)

@show f = bijection(["X1" => "Y1", "X2" => "Y2"])
@show f["X1"] == "Y1"
@show inv(f)["Y1"] == "X1"

@show f = Bijection(["Y1", "Y2"])
@show f[1] == "Y1"
@show inv(f)["Y1"] == 1

println(
  """

  Test CustomVertexGraphs.jl
  """
)

include("CustomVertexGraphs.jl")

g = CustomVertexGraph(grid((4,)), ["A", "B", "C", "D"])
@show g
@show g[["A", "B"]]
@show has_edge(g, "A" => "B")
@show add_edge!(g, "A" => "C")
@show has_edge(g, "A" => "C")

println(
  """

  Test CustomVertexGraphs.jl and SubIndexing.jl
  """
)

include("CustomVertexGraphs.jl")
include("SubIndexing.jl")

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
