abstract type AbstractIndsNetwork{I,V} <: AbstractDataGraph{Vector{I},Vector{I},V,NamedEdge{V}} end

# Field access
data_graph(graph::AbstractIndsNetwork) = _not_implemented()

# AbstractDataGraphs overloads
for f in [:vertex_data, :edge_data]
  @eval begin
    $f(graph::AbstractIndsNetwork, args...) = $f(data_graph(graph), args...)
  end
end
