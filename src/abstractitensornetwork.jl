abstract type AbstractITensorNetwork{V} <: AbstractDataGraph{ITensor,ITensor,V,CustomVertexEdge{V}} end

# Field access
data_graph(graph::AbstractITensorNetwork) = _not_implemented()

# AbstractDataGraphs overloads
for f in [:vertex_data, :edge_data]
  @eval begin
    $f(graph::AbstractITensorNetwork, args...) = $f(data_graph(graph), args...)
  end
end
