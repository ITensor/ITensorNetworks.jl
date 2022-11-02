module ITensorNetworks

using DataGraphs
using Dictionaries
using DocStringExtensions
using Graphs
using ITensors
using ITensors.ContractionSequenceOptimization
using ITensors.ITensorVisualizationCore
using MultiDimDictionaries
using NamedGraphs
using Requires
using SplitApplyCombine
using Suppressor

# TODO: export from ITensors
using ITensors: commontags, @Algorithm_str, Algorithm

using Graphs: AbstractEdge, AbstractGraph, Graph, add_edge!
using MultiDimDictionaries: IndexType, SliceIndex
using NamedGraphs:
  AbstractNamedGraph,
  NamedDimEdge,
  NamedDimGraph,
  parent_graph,
  vertex_to_parent_vertex,
  to_vertex

include("imports.jl")

# General functions
_not_implemented() = error("Not implemented")

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

# When setting an edge with collections of `Index`, set the reverse direction
# edge with the `dag`.
function DataGraphs.reverse_direction(
  is::Union{Index,Tuple{Vararg{<:Index}},Vector{<:Index}}
)
  return dag(is)
end

function Graph(itensors::Vector{ITensor})
  nv_graph = length(itensors)
  graph = Graph(nv_graph)
  for i in 1:(nv_graph - 1), j in (i + 1):nv_graph
    if hascommoninds(itensors[i], itensors[j])
      add_edge!(graph, i => j)
    end
  end
  return graph
end

function NamedDimGraph(itensors::Vector{ITensor})
  return NamedDimGraph(Graph(itensors), 1:length(itensors))
end

front(itr, n=1) = Iterators.take(itr, length(itr) - n)
tail(itr) = Iterators.drop(itr, 1)

# Helper functions
vertex_tag(v::Int) = "$v"

function vertex_tag(v::Tuple)
  t = "$(first(v))"
  for vn in Base.tail(v)
    t *= "×$vn"
  end
  return t
end

# TODO: DELETE
#vertex_tag(v::CartesianKey) = vertex_tag(Tuple(v))

edge_tag(e::Pair) = edge_tag(NamedDimEdge(e))

function edge_tag(e)
  return "$(vertex_tag(src(e)))↔$(vertex_tag(dst(e)))"
end

function vertex_index(v, vertex_space)
  return Index(vertex_space; tags=vertex_tag(v))
end

function edge_index(e, edge_space)
  return Index(edge_space; tags=edge_tag(e))
end

const UniformDataGraph{D} = NamedDimDataGraph{
  D,D,Tuple,NamedDimEdge{Tuple},NamedDimGraph{Tuple}
}

include("utils.jl")
include("namedgraphs.jl")
include("itensors.jl")
include("partition.jl")
include("lattices.jl")
include("abstractindsnetwork.jl")
include("indsnetwork.jl")
include("opsum.jl") # Required IndsNetwork
include("sitetype.jl")
include("abstractitensornetwork.jl")
include("contraction_sequences.jl")
include("apply.jl")
include("expect.jl")
include("models.jl")
include("tebd.jl")
include("itensornetwork.jl")
include(joinpath("treetensornetwork", "treetensornetwork.jl"))

include("exports.jl")

function __init__()
  @require KaHyPar = "2a6221f6-aa48-11e9-3542-2d9e0ef01880" include(
    joinpath("requires", "kahypar.jl")
  )
  @require Metis = "2679e427-3c69-5b7f-982b-ece356f1e94b" include(
    joinpath("requires", "metis.jl")
  )
  @require OMEinsumContractionOrders = "6f22d1fd-8eed-4bb7-9776-e7d684900715" include(
    joinpath("requires", "omeinsumcontractionorders.jl")
  )
end

end
