abstract type AbstractITensorNetwork <: AbstractNamedDimDataGraph{ITensor,ITensor,Tuple,NamedDimEdge{Tuple}} end

# Field access
data_graph(graph::AbstractITensorNetwork) = _not_implemented()

# AbstractDataGraphs overloads
vertex_data(graph::AbstractITensorNetwork, args...) = vertex_data(data_graph(graph), args...)
edge_data(graph::AbstractITensorNetwork, args...) = edge_data(data_graph(graph), args...)
