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

struct BeliefPropagationCache{PTN,MTS} <: AbstractBeliefPropagationCache
  partitioned_tensornetwork::PTN
  messages::MTS
end

#Constructors...
function BeliefPropagationCache(
  ptn::PartitionedGraph; messages=default_messages(ptn)
)
  return BeliefPropagationCache(ptn, messages)
end

function BeliefPropagationCache(tn::AbstractITensorNetwork, partitioned_vertices; kwargs...)
  ptn = PartitionedGraph(tn, partitioned_vertices)
  return BeliefPropagationCache(ptn; kwargs...)
end

function BeliefPropagationCache(
  tn::AbstractITensorNetwork; partitioned_vertices=default_partitioned_vertices(tn), kwargs...
)
  return BeliefPropagationCache(tn, partitioned_vertices; kwargs...)
end

function cache(alg::Algorithm"bp", tn; kwargs...)
  return BeliefPropagationCache(tn; kwargs...)
end

function partitioned_tensornetwork(bp_cache::BeliefPropagationCache)
  return bp_cache.partitioned_tensornetwork
end

messages(bp_cache::BeliefPropagationCache) = bp_cache.messages
function tensornetwork(bp_cache::BeliefPropagationCache)
  return unpartitioned_graph(partitioned_tensornetwork(bp_cache))
end

function default_message(bp_cache::BeliefPropagationCache, edge::PartitionEdge)
  return default_message(scalartype(bp_cache), linkinds(bp_cache, edge))
end

function Base.copy(bp_cache::BeliefPropagationCache)
  return BeliefPropagationCache(
    copy(partitioned_tensornetwork(bp_cache)),
    copy(messages(bp_cache)),
  )
end

function default_bp_maxiter(bp_cache::BeliefPropagationCache)
  return default_bp_maxiter(partitioned_graph(bp_cache))
end
function default_edge_sequence(bp_cache::BeliefPropagationCache)
  return default_edge_sequence(partitioned_tensornetwork(bp_cache))
end

function set_messages(cache::BeliefPropagationCache, messages)
  return BeliefPropagationCache(
    partitioned_tensornetwork(cache), messages
  )
end

function environment(
  bp_cache::BeliefPropagationCache,
  partition_vertices::Vector{<:PartitionVertex};
  ignore_edges=(),
)
  bpes = boundary_partitionedges(bp_cache, partition_vertices; dir=:in)
  ms = messages(bp_cache, setdiff(bpes, ignore_edges))
  return reduce(vcat, ms; init=ITensor[])
end

function region_scalar(
  bp_cache::BeliefPropagationCache,
  pv::PartitionVertex;
  contract_kwargs=(; sequence="automatic"),
)
  incoming_mts = environment(bp_cache, [pv])
  local_state = factor(bp_cache, pv)
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
