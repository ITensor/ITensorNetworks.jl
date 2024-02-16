default_mt_constructor(inds_e) = ITensor[denseblocks(delta(inds_e))]
default_mt_storage(ptn::PartitionedGraph) = Dict()
function default_contractor(contract_list::Vector{ITensor}; kwargs...)
  return contract_exact(contract_list; kwargs...)
end
default_contractor_kwargs() = (; normalize=true, contraction_sequence_alg="optimal")
@traitfn default_bp_niters(g::::(!IsDirected)) = is_tree(g) ? 1 : nothing
@traitfn function default_bp_niters(g::::IsDirected)
  return default_bp_niters(undirected_graph(underlying_graph(g)))
end
function message_tensor_diff(mt_a::Vector{ITensor}, mt_b::Vector{ITensor})
  LHS, RHS = contract(mt_a), contract(mt_b)
  return 0.5 *
         norm((denseblocks(LHS) / sum(diag(LHS))) - (denseblocks(RHS) / sum(diag(RHS))))
end

struct BeliefPropagationCache{PTN,MTS}
  partitioned_itensornetwork::PTN
  message_tensors::MTS
end

#Constructors...
function BeliefPropagationCache(ptn::PartitionedGraph; mt_storage=default_mt_storage)
  mts = default_mt_storage(ptn)
  return BeliefPropagationCache(ptn, mts)
end

function BeliefPropagationCache(tn::ITensorNetwork, partitioned_vertices; kwargs...)
  ptn = PartitionedGraph(tn, partitioned_vertices)
  return BeliefPropagationCache(ptn; kwargs...)
end

partitioned_itensornetwork(bpc::BeliefPropagationCache) = bpc.partitioned_itensornetwork
message_tensors(bpc::BeliefPropagationCache) = bpc.message_tensors
function tensornetwork(bpc::BeliefPropagationCache)
  return unpartitioned_graph(partitioned_itensornetwork(bpc))
end
function NamedGraphs.partitioned_graph(bpc::BeliefPropagationCache)
  return partitioned_graph(partitioned_itensornetwork(bpc))
end

function initial_message_tensor(
  bpc::BeliefPropagationCache, edge::PartitionEdge; mt_constructor=default_mt_constructor
)
  ptn = partitioned_itensornetwork(bpc)
  src_e_itn = subgraph(ptn, src(edge))
  dst_e_itn = subgraph(ptn, dst(edge))
  inds_e = commoninds(src_e_itn, dst_e_itn)
  return mt_constructor(inds_e)
end

function message_tensor(
  bpc::BeliefPropagationCache, edge::PartitionEdge; mt_constructor=default_mt_constructor
)
  mts = message_tensors(bpc)
  return haskey(mts, edge) ? mts[edge] : initial_message_tensor(bpc, edge; mt_constructor)
end
function message_tensors(
  bpc::BeliefPropagationCache, edges::Vector{PartitionEdge}; kwargs...
)
  return [message_tensor(bpc, edge; kwargs...) for edge in edges]
end

function copy(bpc::BeliefPropagationCache)
  return BeliefPropagationCache(
    copy(partitioned_itensornetwork(bpc)), copy(message_tensors(bpc))
  )
end

function default_bp_niters(bpc::BeliefPropagationCache)
  return default_bp_niters(partitioned_graph(bpc))
end
function default_edge_sequence(bpc::BeliefPropagationCache)
  return default_edge_sequence(partitioned_itensornetwork(bpc))
end

"""
Compute message tensor as product of incoming mts and local state
"""
function updated_message_tensor(
  bpc::BeliefPropagationCache,
  edge::PartitionEdge;
  contractor=default_contractor,
  contractor_kwargs=default_contractor_kwargs(),
  mt_constructor=default_mt_constructor,
)
  ptn = partitioned_itensornetwork(bpc)
  pb_edges = partitionedges(ptn, boundary_edges(ptn, vertices(ptn, src(edge)); dir=:in))

  incoming_messages = [
    message_tensor(bpc, e_in; mt_constructor) for e_in in setdiff(pb_edges, [reverse(edge)])
  ]
  incoming_messages = reduce(vcat, incoming_messages; init=ITensor[])

  contract_list = ITensor[
    incoming_messages
    Vector{ITensor}(subgraph(ptn, src(edge)))
  ]

  return contractor(contract_list; contractor_kwargs...)
end

"""
Do a sequential update of the message tensors on `edges`
"""
function update(
  bpc::BeliefPropagationCache,
  edges::Vector{<:PartitionEdge};
  compute_norm=false,
  mt_constructor=default_mt_constructor,
  kwargs...,
)
  bpc_updated = copy(bpc)
  mts = message_tensors(bpc_updated)
  c = 0
  for e in edges
    mts[e] = updated_message_tensor(bpc_updated, e; mt_constructor, kwargs...)
    if compute_norm
      c += message_tensor_diff(message_tensor(bpc, e; mt_constructor), mts[e])
    end
  end
  return bpc_updated, c / (length(edges))
end

"""
Update the message tensor on a single edge
"""
function update(bpc::BeliefPropagationCache, edge::PartitionEdge; kwargs...)
  return update(bpc, [edge]; kwargs...)
end

"""
Do parallel updates between groups of edges of all message tensors
Currently we send the full message tensor data struct to update for each edge_group. But really we only need the
mts relevant to that group.
"""
function update(
  bpc::BeliefPropagationCache, edge_groups::Vector{<:Vector{<:PartitionEdge}}; kwargs...
)
  new_mts = copy(message_tensors(bpc))
  c = 0
  for edges in edge_groups
    bpc_t, ct = update(bpc, edges; kwargs...)
    for e in edges
      new_mts[e] = message_tensor(bpc_t, e)
    end
    c += ct
  end
  return BeliefPropagationCache(copy(partitioned_itensornetwork(bpc)), new_mts),
  c / (length(edge_groups))
end

"""
More generic interface for update, with default params
"""
function update(
  bpc::BeliefPropagationCache;
  edges=default_edge_sequence(bpc),
  niters=default_bp_niters(bpc),
  target_precision=nothing,
  verbose=false,
  kwargs...,
)
  compute_norm = !isnothing(target_precision)
  if isnothing(niters)
    error("You need to specify a number of iterations for BP!")
  end
  for i in 1:niters
    bpc, c = update(bpc, edges; compute_norm, kwargs...)
    if compute_norm && c <= target_precision
      if verbose
        println("BP converged to desired precision after $i iterations.")
      end
      break
    end
  end
  return bpc
end

"""
Update the tensornetwork inside the cache
"""
function update(bpc::BeliefPropagationCache, states::Vector{ITensor}, vertices::Vector)
  bpc = copy(bpc)
  tn = tensornetwork(bpc)

  for (state, vertex) in zip(states, vertices)
    setindex_preserve_graph!(tn, state, vertex)
  end
  return bpc
end

function update(bpc, state, vertex)
  return update(bpc, ITensor[state], [vertex])
end

"""
Get the relevant message tensors coming on to the vertices (of the underlying itensornetwork in the cache)
"""
function environment_tensors(bpc::BeliefPropagationCache, verts::Vector)
  ptn = partitioned_itensornetwork(bpc)
  partition_verts = partitionvertices(ptn, verts)
  central_verts = vertices(ptn, partition_verts)

  pedges = partitionedges(ptn, boundary_edges(ptn, central_verts; dir=:in))
  env_tensors = [message_tensor(bpc, e) for e in pedges]
  env_tensors = reduce(vcat, env_tensors; init=ITensor[])
  central_tensors = ITensor[
    (unpartitioned_graph(ptn))[v] for v in setdiff(central_verts, verts)
  ]

  return vcat(env_tensors, central_tensors)
end

function environment_tensors(
  bpc::BeliefPropagationCache, partition_verts::Vector{<:PartitionVertex}
)
  return environment_tensors(
    bpc, vertices(partitioned_itensornetwork(bpc), partition_verts)
  )
end
