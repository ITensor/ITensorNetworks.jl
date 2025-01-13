using Graphs: IsDirected
using SplitApplyCombine: group
using LinearAlgebra: diag, dot
using ITensors: dir
using ITensorMPS: ITensorMPS
using NamedGraphs.PartitionedGraphs:
  PartitionedGraphs,
  PartitionedGraph,
  PartitionVertex,
  boundary_partitionedges,
  partitionvertices,
  partitionedges,
  unpartitioned_graph
using SimpleTraits: SimpleTraits, Not, @traitfn
using NDTensors: NDTensors

function default_cache_construction_kwargs(alg::Algorithm"bp", ψ::AbstractITensorNetwork)
  return (; partitioned_vertices=default_partitioned_vertices(ψ))
end

function default_cache_construction_kwargs(alg::Algorithm"bp", pg::PartitionedGraph)
  return (;)
end

struct BeliefPropagationCache{PTN,MTS} <: AbstractBeliefPropagationCache
  partitioned_tensornetwork::PTN
  messages::MTS
end

#Constructors...
function BeliefPropagationCache(ptn::PartitionedGraph; messages=default_messages(ptn))
  return BeliefPropagationCache(ptn, messages)
end

function BeliefPropagationCache(tn::AbstractITensorNetwork, partitioned_vertices; kwargs...)
  ptn = PartitionedGraph(tn, partitioned_vertices)
  return BeliefPropagationCache(ptn; kwargs...)
end

function BeliefPropagationCache(
  tn::AbstractITensorNetwork;
  partitioned_vertices=default_partitioned_vertices(tn),
  kwargs...,
)
  return BeliefPropagationCache(tn, partitioned_vertices; kwargs...)
end

function cache(alg::Algorithm"bp", tn; kwargs...)
  return BeliefPropagationCache(tn; kwargs...)
end
default_cache_update_kwargs(alg::Algorithm"bp") = (; maxiter=25, tol=1e-8)

function partitioned_tensornetwork(bp_cache::BeliefPropagationCache)
  return bp_cache.partitioned_tensornetwork
end

messages(bp_cache::BeliefPropagationCache) = bp_cache.messages

function default_message(bp_cache::BeliefPropagationCache, edge::PartitionEdge)
  return default_message(scalartype(bp_cache), linkinds(bp_cache, edge))
end

function Base.copy(bp_cache::BeliefPropagationCache)
  return BeliefPropagationCache(
    copy(partitioned_tensornetwork(bp_cache)), copy(messages(bp_cache))
  )
end

default_message_update_alg(bp_cache::BeliefPropagationCache) = "simplebp"

function default_bp_maxiter(alg::Algorithm"simplebp", bp_cache::BeliefPropagationCache)
  return default_bp_maxiter(partitioned_graph(bp_cache))
end
function default_edge_sequence(alg::Algorithm"simplebp", bp_cache::BeliefPropagationCache)
  return default_edge_sequence(partitioned_tensornetwork(bp_cache))
end
function default_message_update_kwargs(
  alg::Algorithm"simplebp", bpc::AbstractBeliefPropagationCache
)
  return (;)
end

partitions(bpc::BeliefPropagationCache) = partitionvertices(partitioned_tensornetwork(bpc))
partitionpairs(bpc::BeliefPropagationCache) = partitionedges(partitioned_tensornetwork(bpc))

function set_messages(cache::BeliefPropagationCache, messages)
  return BeliefPropagationCache(partitioned_tensornetwork(cache), messages)
end

function environment(bpc::BeliefPropagationCache, verts::Vector; kwargs...)
  partition_verts = partitionvertices(bpc, verts)
  messages = incoming_messages(bpc, partition_verts; kwargs...)
  central_tensors = factors(bpc, setdiff(vertices(bpc, partition_verts), verts))
  return vcat(messages, central_tensors)
end

function region_scalar(
  bp_cache::BeliefPropagationCache,
  pv::PartitionVertex;
  contract_kwargs=(; sequence="automatic"),
)
  incoming_mts = incoming_messages(bp_cache, [pv])
  local_state = factors(bp_cache, pv)
  return contract(vcat(incoming_mts, local_state); contract_kwargs...)[]
end

function region_scalar(
  bp_cache::BeliefPropagationCache,
  pe::PartitionEdge;
  contract_kwargs=(; sequence="automatic"),
)
  return contract(
    vcat(message(bp_cache, pe), message(bp_cache, reverse(pe))); contract_kwargs...
  )[]
end
