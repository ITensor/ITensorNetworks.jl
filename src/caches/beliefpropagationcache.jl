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

default_message(elt, inds_e) = ITensor[denseblocks(delta(elt, i)) for i in inds_e]
default_messages(ptn::PartitionedGraph) = Dictionary()
function default_message_update(contract_list::Vector{ITensor}; normalize=true, kwargs...)
  sequence = optimal_contraction_sequence(contract_list)
  updated_messages = contract(contract_list; sequence, kwargs...)
  message_norm = norm(updated_messages)
  if normalize && !iszero(message_norm)
    updated_messages /= message_norm
  end
  return ITensor[updated_messages]
end
@traitfn default_bp_maxiter(g::::(!IsDirected)) = is_tree(g) ? 1 : nothing
@traitfn function default_bp_maxiter(g::::IsDirected)
  return default_bp_maxiter(undirected_graph(underlying_graph(g)))
end
default_partitioned_vertices(ψ::AbstractITensorNetwork) = group(v -> v, vertices(ψ))
function default_partitioned_vertices(f::AbstractFormNetwork)
  return group(v -> original_state_vertex(f, v), vertices(f))
end
default_cache_update_kwargs(cache) = (; maxiter=25, tol=1e-8)
function default_cache_construction_kwargs(alg::Algorithm"bp", ψ::AbstractITensorNetwork)
  return (; partitioned_vertices=default_partitioned_vertices(ψ))
end

#TODO: Take `dot` without precontracting the messages to allow scaling to more complex messages
function message_diff(message_a::Vector{ITensor}, message_b::Vector{ITensor})
  lhs, rhs = contract(message_a), contract(message_b)
  f = abs2(dot(lhs / norm(lhs), rhs / norm(rhs)))
  return 1 - f
end

struct BeliefPropagationCache{PTN,MTS,DM}
  partitioned_tensornetwork::PTN
  messages::MTS
  default_message::DM
end

#Constructors...
function BeliefPropagationCache(
  ptn::PartitionedGraph; messages=default_messages(ptn), default_message=default_message
)
  return BeliefPropagationCache(ptn, messages, default_message)
end

function BeliefPropagationCache(tn, partitioned_vertices; kwargs...)
  ptn = PartitionedGraph(tn, partitioned_vertices)
  return BeliefPropagationCache(ptn; kwargs...)
end

function BeliefPropagationCache(
  tn; partitioned_vertices=default_partitioned_vertices(tn), kwargs...
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
default_message(bp_cache::BeliefPropagationCache) = bp_cache.default_message
function tensornetwork(bp_cache::BeliefPropagationCache)
  return unpartitioned_graph(partitioned_tensornetwork(bp_cache))
end

#Forward from partitioned graph
for f in [
  :(PartitionedGraphs.partitioned_graph),
  :(PartitionedGraphs.partitionedge),
  :(PartitionedGraphs.partitionvertices),
  :(PartitionedGraphs.vertices),
  :(PartitionedGraphs.boundary_partitionedges),
  :(ITensorMPS.linkinds),
]
  @eval begin
    function $f(bp_cache::BeliefPropagationCache, args...; kwargs...)
      return $f(partitioned_tensornetwork(bp_cache), args...; kwargs...)
    end
  end
end

NDTensors.scalartype(bp_cache) = scalartype(tensornetwork(bp_cache))

function default_message(bp_cache::BeliefPropagationCache, edge::PartitionEdge)
  return default_message(bp_cache)(scalartype(bp_cache), linkinds(bp_cache, edge))
end

function message(bp_cache::BeliefPropagationCache, edge::PartitionEdge)
  mts = messages(bp_cache)
  return get(() -> default_message(bp_cache, edge), mts, edge)
end
function messages(bp_cache::BeliefPropagationCache, edges; kwargs...)
  return map(edge -> message(bp_cache, edge; kwargs...), edges)
end

function Base.copy(bp_cache::BeliefPropagationCache)
  return BeliefPropagationCache(
    copy(partitioned_tensornetwork(bp_cache)),
    copy(messages(bp_cache)),
    default_message(bp_cache),
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
    partitioned_tensornetwork(cache), messages, default_message(cache)
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

function environment(
  bp_cache::BeliefPropagationCache, partition_vertex::PartitionVertex; kwargs...
)
  return environment(bp_cache, [partition_vertex]; kwargs...)
end

function environment(bp_cache::BeliefPropagationCache, verts::Vector)
  partition_verts = partitionvertices(bp_cache, verts)
  messages = environment(bp_cache, partition_verts)
  central_tensors = factors(bp_cache, setdiff(vertices(bp_cache, partition_verts), verts))
  return vcat(messages, central_tensors)
end

function factors(bp_cache::BeliefPropagationCache, verts::Vector)
  return ITensor[tensornetwork(bp_cache)[v] for v in verts]
end

function factor(bp_cache::BeliefPropagationCache, vertex::PartitionVertex)
  return factors(bp_cache, vertices(bp_cache, vertex))
end

"""
Compute message tensor as product of incoming mts and local state
"""
function update_message(
  bp_cache::BeliefPropagationCache,
  edge::PartitionEdge;
  message_update=default_message_update,
  message_update_kwargs=(;),
)
  vertex = src(edge)
  messages = environment(bp_cache, vertex; ignore_edges=PartitionEdge[reverse(edge)])
  state = factor(bp_cache, vertex)

  return message_update(ITensor[messages; state]; message_update_kwargs...)
end

"""
Do a sequential update of the message tensors on `edges`
"""
function update(
  bp_cache::BeliefPropagationCache,
  edges::Vector{<:PartitionEdge};
  (update_diff!)=nothing,
  kwargs...,
)
  bp_cache_updated = copy(bp_cache)
  mts = messages(bp_cache_updated)
  for e in edges
    set!(mts, e, update_message(bp_cache_updated, e; kwargs...))
    if !isnothing(update_diff!)
      update_diff![] += message_diff(message(bp_cache, e), mts[e])
    end
  end
  return bp_cache_updated
end

"""
Update the message tensor on a single edge
"""
function update(bp_cache::BeliefPropagationCache, edge::PartitionEdge; kwargs...)
  return update(bp_cache, [edge]; kwargs...)
end

"""
Do parallel updates between groups of edges of all message tensors
Currently we send the full message tensor data struct to update for each edge_group. But really we only need the
mts relevant to that group.
"""
function update(
  bp_cache::BeliefPropagationCache,
  edge_groups::Vector{<:Vector{<:PartitionEdge}};
  kwargs...,
)
  new_mts = copy(messages(bp_cache))
  for edges in edge_groups
    bp_cache_t = update(bp_cache, edges; kwargs...)
    for e in edges
      new_mts[e] = message(bp_cache_t, e)
    end
  end
  return set_messages(bp_cache, new_mts)
end

"""
More generic interface for update, with default params
"""
function update(
  bp_cache::BeliefPropagationCache;
  edges=default_edge_sequence(bp_cache),
  maxiter=default_bp_maxiter(bp_cache),
  tol=nothing,
  verbose=false,
  kwargs...,
)
  compute_error = !isnothing(tol)
  if isnothing(maxiter)
    error("You need to specify a number of iterations for BP!")
  end
  for i in 1:maxiter
    diff = compute_error ? Ref(0.0) : nothing
    bp_cache = update(bp_cache, edges; (update_diff!)=diff, kwargs...)
    if compute_error && (diff.x / length(edges)) <= tol
      if verbose
        println("BP converged to desired precision after $i iterations.")
      end
      break
    end
  end
  return bp_cache
end

"""
Update the tensornetwork inside the cache
"""
function update_factors(bp_cache::BeliefPropagationCache, factors)
  bp_cache = copy(bp_cache)
  tn = tensornetwork(bp_cache)
  for vertex in eachindex(factors)
    # TODO: Add a check that this preserves the graph structure.
    setindex_preserve_graph!(tn, factors[vertex], vertex)
  end
  return bp_cache
end

function update_factor(bp_cache, vertex, factor)
  return update_factors(bp_cache, Dictionary([vertex], [factor]))
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

function vertex_scalars(
  bp_cache::BeliefPropagationCache,
  pvs=partitionvertices(partitioned_tensornetwork(bp_cache));
  kwargs...,
)
  return map(pv -> region_scalar(bp_cache, pv; kwargs...), pvs)
end

function edge_scalars(
  bp_cache::BeliefPropagationCache,
  pes=partitionedges(partitioned_tensornetwork(bp_cache));
  kwargs...,
)
  return map(pe -> region_scalar(bp_cache, pe; kwargs...), pes)
end

function scalar_factors_quotient(bp_cache::BeliefPropagationCache)
  return vertex_scalars(bp_cache), edge_scalars(bp_cache)
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
