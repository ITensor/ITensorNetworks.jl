# ITensorVisualizationBase overload
function visualize(
  graph::AbstractNamedGraph,
  args...;
  vertex_labels_prefix=nothing,
  vertex_labels=nothing,
  kwargs...,
)
  if !isnothing(vertex_labels_prefix)
    vertex_labels = [vertex_labels_prefix * string(v) for v in vertices(graph)]
  end
  #edge_labels = [string(e) for e in edges(graph)]
  return visualize(parent_graph(graph), args...; vertex_labels, kwargs...)
end

# ITensorVisualizationBase overload
function visualize(graph::AbstractDataGraph, args...; kwargs...)
  return visualize(underlying_graph(graph), args...; kwargs...)
end
