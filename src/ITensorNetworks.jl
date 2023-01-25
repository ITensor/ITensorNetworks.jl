module ITensorNetworks

using AbstractTrees
using Compat
using DataGraphs
using Dictionaries
using DocStringExtensions
using Graphs
using Graphs.SimpleGraphs # AbstractSimpleGraph
using IsApprox
using ITensors
using ITensors.ContractionSequenceOptimization
using ITensors.ITensorVisualizationCore
using ITensors.LazyApply
using IterTools
using KrylovKit: KrylovKit
using NamedGraphs
using Observers
using Printf
using Requires
using SimpleTraits
using SparseArrayKit
using SplitApplyCombine
using StaticArrays
using Suppressor
using TimerOutputs

using DataGraphs: IsUnderlyingGraph, edge_data_type, vertex_data_type
using Graphs: AbstractEdge, AbstractGraph, Graph, add_edge!
using ITensors:
  @Algorithm_str,
  @debug_check,
  @timeit_debug,
  AbstractMPS,
  Algorithm,
  OneITensor,
  check_hascommoninds,
  commontags,
  orthocenter,
  ProjMPS,
  set_nsite!
using KrylovKit: exponentiate, eigsolve, linsolve
using NamedGraphs:
  AbstractNamedGraph,
  parent_graph,
  vertex_to_parent_vertex,
  parent_vertices_to_vertices,
  not_implemented

include("imports.jl")

# TODO: Move to `DataGraphs.jl`
edge_data_type(::AbstractNamedGraph) = Any
isassigned(::AbstractNamedGraph, ::Any) = false
function iterate(::AbstractDataGraph)
  return error(
    "Iterating data graphs is not yet defined. We may define it in the future as iterating through the vertex and edge data.",
  )
end

include("observers.jl")
include("utils.jl")
include("visualize.jl")
include("graphs.jl")
include("itensors.jl")
include("partition.jl")
include("lattices.jl")
include("abstractindsnetwork.jl")
include("indextags.jl")
include("indsnetwork.jl")
include("opsum.jl")
include("sitetype.jl")
include("abstractitensornetwork.jl")
include("contraction_sequences.jl")
include("apply.jl")
include("expect.jl")
include("models.jl")
include("tebd.jl")
include("itensornetwork.jl")
include("utility.jl")
include("specialitensornetworks.jl")
include("renameitensornetwork.jl")
include("boundarymps.jl")
include("beliefpropagation.jl")
include("contraction_tree_to_graph.jl")
include(joinpath("Graphs", "abstractgraph.jl"))
include(joinpath("treetensornetworks", "abstracttreetensornetwork.jl"))
include(joinpath("treetensornetworks", "ttn.jl"))
include(joinpath("treetensornetworks", "opsum_to_ttn.jl"))
include(joinpath("treetensornetworks", "projttns", "abstractprojttn.jl"))
include(joinpath("treetensornetworks", "projttns", "projttn.jl"))
include(joinpath("treetensornetworks", "projttns", "projttnsum.jl"))
include(joinpath("treetensornetworks", "projttns", "projttn_apply.jl"))
include(joinpath("treetensornetworks", "solvers", "solver_utils.jl"))
include(joinpath("treetensornetworks", "solvers", "applyexp.jl"))
include(joinpath("treetensornetworks", "solvers", "tdvporder.jl"))
include(joinpath("treetensornetworks", "solvers", "tdvpinfo.jl"))
include(joinpath("treetensornetworks", "solvers", "tdvp_step.jl"))
include(joinpath("treetensornetworks", "solvers", "tdvp_generic.jl"))
include(joinpath("treetensornetworks", "solvers", "tdvp.jl"))
include(joinpath("treetensornetworks", "solvers", "dmrg.jl"))
include(joinpath("treetensornetworks", "solvers", "dmrg_x.jl"))
include(joinpath("treetensornetworks", "solvers", "contract.jl"))
include(joinpath("treetensornetworks", "solvers", "linsolve.jl"))
include(joinpath("treetensornetworks", "solvers", "tree_sweeping.jl"))

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
