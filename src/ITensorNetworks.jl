module ITensorNetworks

using Compat
using DataGraphs
using Dictionaries
using DocStringExtensions
using Graphs
using IsApprox
using ITensors
using ITensors.ContractionSequenceOptimization
using ITensors.ITensorVisualizationCore
using NamedGraphs
using Requires
using SimpleTraits
using SparseArrayKit
using SplitApplyCombine
using StaticArrays
using Suppressor

# TODO: export from ITensors
using ITensors: commontags, @Algorithm_str, Algorithm, OneITensor

using Graphs: AbstractEdge, AbstractGraph, Graph, add_edge!
using NamedGraphs:
  AbstractNamedGraph,
  parent_graph,
  vertex_to_parent_vertex,
  not_implemented
using DataGraphs: vertex_data_type

include("imports.jl")

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
include(joinpath("treetensornetwork", "abstracttreetensornetwork.jl"))
include(joinpath("treetensornetwork", "ttns.jl"))
include(joinpath("treetensornetwork", "ttno.jl"))
include(joinpath("treetensornetwork", "opsum_to_ttno.jl"))
include(joinpath("treetensornetwork", "abstractprojttno.jl"))
include(joinpath("treetensornetwork", "projttno.jl"))
include(joinpath("treetensornetwork", "projttnosum.jl"))
include("utility.jl")
include("specialitensornetworks.jl")
include("renameitensornetwork.jl")
include("boundarymps.jl")
include("subgraphs.jl")
include("beliefpropagation.jl")

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
