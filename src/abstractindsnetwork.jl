abstract type AbstractIndsNetwork{I} <: AbstractNamedDimDataGraph{Vector{I},Vector{I},Tuple,NamedDimEdge{Tuple}} end

# Field access
data_graph(graph::AbstractIndsNetwork) = _not_implemented()

# AbstractDataGraphs overloads
vertex_data(graph::AbstractIndsNetwork, args...) = vertex_data(data_graph(graph), args...)
edge_data(graph::AbstractIndsNetwork, args...) = edge_data(data_graph(graph), args...)
