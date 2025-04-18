using Graphs: IsDirected
using SplitApplyCombine: group
using LinearAlgebra: diag, dot
using ITensors: dir
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

default_message_update_alg(bp_cache::BeliefPropagationCache) = "bp"

function default_bp_maxiter(alg::Algorithm"bp", bp_cache::BeliefPropagationCache)
  return default_bp_maxiter(partitioned_graph(bp_cache))
end
function default_edge_sequence(alg::Algorithm"bp", bp_cache::BeliefPropagationCache)
  return default_edge_sequence(partitioned_tensornetwork(bp_cache))
end
function default_message_update_kwargs(
  alg::Algorithm"bp", bpc::AbstractBeliefPropagationCache
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

function region_scalar(bp_cache::BeliefPropagationCache, pv::PartitionVertex)
  incoming_mts = incoming_messages(bp_cache, [pv])
  local_state = factors(bp_cache, pv)
  ts = vcat(incoming_mts, local_state)
  sequence = contraction_sequence(ts; alg="optimal")
  return contract(ts; sequence)[]
end

function region_scalar(bp_cache::BeliefPropagationCache, pe::PartitionEdge)
  ts = vcat(message(bp_cache, pe), message(bp_cache, reverse(pe)))
  sequence = contraction_sequence(ts; alg="optimal")
  return contract(ts; sequence)[]
end

function normalize_messages(bp_cache::BeliefPropagationCache, pes::Vector{<:PartitionEdge})
  bp_cache = copy(bp_cache)
  mts = messages(bp_cache)
  for pe in pes
    me, mer = only(mts[pe]), only(mts[reverse(pe)])
    me, mer = normalize(me), normalize(mer)
    n = dot(me, mer)
    if isreal(n) && n < 0
      set!(mts, pe, ITensor[(sgn(n) / sqrt(abs(n))) * me])
      set!(mts, reverse(pe), ITensor[(1 / sqrt(abs(n))) * mer])
    else
      set!(mts, pe, ITensor[(1 / sqrt(n)) * me])
      set!(mts, reverse(pe), ITensor[(1 / sqrt(n)) * mer])
    end
  end
  return bp_cache
end

function normalize_message(bp_cache::BeliefPropagationCache, pe::PartitionEdge)
  return normalize_messages(bp_cache, PartitionEdge[pe])
end

function normalize_messages(bp_cache::BeliefPropagationCache)
  return normalize_messages(bp_cache, partitionedges(partitioned_tensornetwork(bp_cache)))
end
