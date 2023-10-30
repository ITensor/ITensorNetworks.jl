module ITensorNetworks

using AbstractTrees
using Accessors
using Combinatorics
using Compat
using DataGraphs
using DataStructures
using Dictionaries
using Distributions
using DocStringExtensions
using Graphs
using GraphsFlows
using Graphs.SimpleGraphs # AbstractSimpleGraph
using IsApprox
using ITensors
using ITensors.ContractionSequenceOptimization
using ITensors.ITensorVisualizationCore
using ITensors.LazyApply
using IterTools
using KrylovKit: KrylovKit
using LinearAlgebra
using NamedGraphs
using Observers
using Observers.DataFrames: select!
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
  δ,
  AbstractMPS,
  Algorithm,
  OneITensor,
  check_hascommoninds,
  commontags,
  dim,
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
include("mincut.jl")
include("contract_deltas.jl")
include("binary_tree_partition.jl")
include("approx_itensornetwork/utils.jl")
include("approx_itensornetwork/density_matrix.jl")
include("approx_itensornetwork/ttn_svd.jl")
include("approx_itensornetwork/approx_itensornetwork.jl")
include("contract.jl")
include("utility.jl")
include("specialitensornetworks.jl")
include("renameitensornetwork.jl")
include("boundarymps.jl")
include("beliefpropagation/beliefpropagation.jl")
include("beliefpropagation/sqrt_beliefpropagation.jl")
include("contraction_tree_to_graph.jl")
include("gauging.jl")
include("utils.jl")
include("tensornetworkoperators.jl")
include("ITensorsExt/itensorutils.jl")
include("Graphs/abstractgraph.jl")
include("Graphs/abstractdatagraph.jl")
include("treetensornetworks/abstracttreetensornetwork.jl")
include("treetensornetworks/ttn.jl")
include("treetensornetworks/opsum_to_ttn.jl")
include("treetensornetworks/projttns/abstractprojttn.jl")
include("treetensornetworks/projttns/projttn.jl")
include("treetensornetworks/projttns/projttnsum.jl")
include("treetensornetworks/projttns/projttn_apply.jl")
include("treetensornetworks/solvers/solver_utils.jl")
include("treetensornetworks/solvers/applyexp.jl")
include("treetensornetworks/solvers/update_step.jl")
include("treetensornetworks/solvers/alternating_update.jl")
include("treetensornetworks/solvers/tdvp.jl")
include("treetensornetworks/solvers/dmrg.jl")
include("treetensornetworks/solvers/dmrg_x.jl")
include("treetensornetworks/solvers/contract.jl")
include("treetensornetworks/solvers/linsolve.jl")
include("treetensornetworks/solvers/tree_sweeping.jl")
include("itensornetworkcache/problems.jl")
include("itensornetworkcache/abstractitensornetworkcache.jl")
include("itensornetworkcache/rayleighquotientcache.jl")
include("itensornetworkcache/quadraticform.jl")
include("itensornetworkcache/bpcache.jl")
include("itensornetworkcache/alternating_update/alternating_update.jl")

include("exports.jl")

function __init__()
  @require KaHyPar = "2a6221f6-aa48-11e9-3542-2d9e0ef01880" include(
    "requires/kahypar.jl"
  )
  @require Metis = "2679e427-3c69-5b7f-982b-ece356f1e94b" include(
    "requires/metis.jl"
  )
  @require OMEinsumContractionOrders = "6f22d1fd-8eed-4bb7-9776-e7d684900715" include(
    "requires/omeinsumcontractionorders.jl"
  )
end

end
