# TODO: Move to `NamedGraphsITensorVisualizationCoreExt`.
using Graphs: vertices
using NamedGraphs: NamedGraphs, AbstractNamedGraph
using ITensors.ITensorVisualizationCore: ITensorVisualizationCore
function ITensorVisualizationCore.visualize(
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
  return ITensorVisualizationCore.visualize(
    NamedGraphs.position_graph(graph), args...; vertex_labels, kwargs...
  )
end

# TODO: Move to `DataGraphsITensorVisualizationCoreExt`.
using DataGraphs: DataGraphs, AbstractDataGraph
using ITensors.ITensorVisualizationCore: ITensorVisualizationCore
function ITensorVisualizationCore.visualize(graph::AbstractDataGraph, args...; kwargs...)
  return ITensorVisualizationCore.visualize(
    DataGraphs.underlying_graph(graph), args...; kwargs...
  )
end
