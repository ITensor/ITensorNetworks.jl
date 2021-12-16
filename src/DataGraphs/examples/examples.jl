
include(joinpath("..", "src", "SimpleMetaGraphs.jl"))

g = grid((4,))
vertex_metadata = dictionary([1 => "V1", 2 => "V2", 3 => "V3", 4 => "V4"])
edge_metadata = dictionary([Edge(1, 2) => "E12"])
mg = SimpleMetaGraph(g, vertex_metadata, edge_metadata)

@show mg[1] == "V1"
@show mg[2] == "V2"
@show mg[3] == "V3"
@show mg[4] == "V4"

@show mg[1 => 2] == "E12"
@show mg[1, 2] == "E12"
@show mg[Edge(1, 2)] == "E12"

mg[1 => 4] = "E14"
@show mg[1 => 4]
