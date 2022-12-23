module ITensorNetworks

using Compat
using DataGraphs
using Dictionaries
using DocStringExtensions
using Graphs
using ITensors
using ITensors.ContractionSequenceOptimization
using ITensors.ITensorVisualizationCore
using NamedGraphs
using Requires
using SimpleTraits
using SplitApplyCombine
using Suppressor

# TODO: export from ITensors
using ITensors: commontags, @Algorithm_str, Algorithm
using Graphs: AbstractEdge, AbstractGraph, Graph, add_edge!
using Graphs.SimpleGraphs # AbstractSimpleGraph
using NamedGraphs:
  AbstractNamedGraph,
  parent_graph,
  vertex_to_parent_vertex,
  parent_vertices_to_vertices,
  not_implemented
using DataGraphs: edge_data_type, vertex_data_type

include("imports.jl")

# TODO: Move to `DataGraphs.jl`
edge_data_type(::AbstractNamedGraph) = Any
isassigned(::AbstractNamedGraph, ::Any) = false
function iterate(::AbstractDataGraph)
  return error(
    "Iterating data graphs is not yet defined. We may define it in the future as iterating through the vertex and edge data.",
  )
end

include("utils.jl")
include("visualize.jl")
include("graphs.jl")
include("itensors.jl")
include("partition.jl")
include("lattices.jl")
include("abstractindsnetwork.jl")
include("indextags.jl")
include("indsnetwork.jl")
include("opsum.jl") # Requires IndsNetwork
include("sitetype.jl")
include("abstractitensornetwork.jl")
include("contraction_sequences.jl")
include("apply.jl")
include("expect.jl")
include("models.jl")
include("tebd.jl")
include("itensornetwork.jl")
include("specialitensornetworks.jl")
include("renameitensornetwork.jl")
include("boundarymps.jl")
include("beliefpropagation.jl")
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
