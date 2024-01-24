default_mt_constructor(inds_e) = ITensor[denseblocks(delta(inds_e))]
default_bp_cache(ptn::PartitionedGraph) = Dict()
function default_contractor(contract_list::Vector{ITensor})
  return ITensor[normalize!(
    contract(contract_list; sequence=contraction_sequence(contract_list; alg="optimal"))
  )]
end

function contract_to_MPS(contract_list::Vector{ITensor}; svd_kwargs...)
  contract_kwargs = (;
    alg="density_matrix",
    output_structure=path_graph_structure,
    contraction_sequence_alg="optimal",
    svd_kwargs...,
  )

  tn, _ = contract(ITensorNetwork(contract_list); contract_kwargs...)
  mts = Vector{ITensor}(tn)
  mts = normalize!.(mts)
  return mts
end

function message_tensor(
  ptn::PartitionedGraph, edge::PartitionEdge; mt_constructor=default_mt_constructor
)
  src_e_itn = subgraph(ptn, src(edge))
  dst_e_itn = subgraph(ptn, dst(edge))
  inds_e = commoninds(src_e_itn, dst_e_itn)
  return mt_constructor(inds_e)
end

"""
Do a single update of a message tensor using the current subgraph and the incoming mts
"""
function update_message_tensor(
  ptn::PartitionedGraph,
  edge::PartitionEdge,
  mts;
  contractor=default_contractor,
  mt_constructor=default_mt_constructor,
)
  pedges = setdiff(
    partitionedges(ptn, boundary_edges(ptn, vertices(ptn, src(edge)); dir=:in)),
    [reverse(edge)],
  )

  incoming_messages = [
    e_in ∈ keys(mts) ? mts[e_in] : message_tensor(ptn, e_in; mt_constructor) for
    e_in in pedges
  ]
  incoming_messages = reduce(vcat, incoming_messages; init=ITensor[])

  contract_list = ITensor[
    incoming_messages
    Vector{ITensor}(subgraph(ptn, src(edge)))
  ]

  return contractor(contract_list)
end

"""
Do a sequential update of message tensors on `edges` for a given ITensornetwork and its partition into sub graphs
"""
function belief_propagation_iteration(
  ptn::PartitionedGraph,
  mts,
  edges::Vector{<:PartitionEdge};
  contractor=default_contractor,
  compute_norm=false,
)
  new_mts = copy(mts)
  c = 0
  for e in edges
    new_mts[e] = update_message_tensor(ptn, e, new_mts; contractor)

    if compute_norm
      LHS = e ∈ keys(mts) ? contract(mts[e]) : contract(message_tensor(ptn, e))
      RHS = contract(new_mts[e])
      #This line only makes sense if the message tensors are rank 2??? Should fix this.
      LHS /= sum(diag(LHS))
      RHS /= sum(diag(RHS))
      c += 0.5 * norm(denseblocks(LHS) - denseblocks(RHS))
    end
  end
  return new_mts, c / (length(edges))
end

"""
Do parallel updates between groups of edges of all message tensors for a given ITensornetwork and its partition into sub graphs.
Currently we send the full message tensor data struct to belief_propagation_iteration for each subgraph. But really we only need the
mts relevant to that subgraph.
"""
function belief_propagation_iteration(
  ptn::PartitionedGraph,
  mts,
  edge_groups::Vector{<:Vector{<:PartitionEdge}};
  contractor=default_contractor,
  compute_norm=false,
)
  new_mts = copy(mts)
  c = 0
  for edges in edge_groups
    updated_mts, ct = belief_propagation_iteration(
      ptn, mts, edges; contractor, compute_norm
    )
    for e in edges
      new_mts[e] = updated_mts[e]
    end
    c += ct
  end
  return new_mts, c / (length(edge_groups))
end

function belief_propagation_iteration(
  ptn::PartitionedGraph,
  mts;
  contractor=default_contractor,
  compute_norm=false,
  edges=PartitionEdge.(edge_sequence(partitioned_graph(ptn))),
)
  return belief_propagation_iteration(ptn, mts, edges; contractor, compute_norm)
end

function belief_propagation(
  ptn::PartitionedGraph,
  mts;
  contractor=default_contractor,
  niters=default_bp_niters(partitioned_graph(ptn)),
  target_precision=nothing,
  edges=PartitionEdge.(edge_sequence(partitioned_graph(ptn))),
  verbose=false,
)
  compute_norm = !isnothing(target_precision)
  if isnothing(niters)
    error("You need to specify a number of iterations for BP!")
  end
  for i in 1:niters
    mts, c = belief_propagation_iteration(ptn, mts, edges; contractor, compute_norm)
    if compute_norm && c <= target_precision
      if verbose
        println("BP converged to desired precision after $i iterations.")
      end
      break
    end
  end
  return mts
end

function belief_propagation(ptn::PartitionedGraph; bp_cache=default_bp_cache, kwargs...)
  mts = bp_cache(ptn)
  return belief_propagation(ptn, mts; kwargs...)
end
"""
Given a subet of partitionvertices of a ptn get the incoming message tensors to that region
"""
function environment_tensors(ptn::PartitionedGraph, mts, verts::Vector)
  partition_verts = partitionvertices(ptn, verts)
  central_verts = vertices(ptn, partition_verts)

  pedges = partitionedges(ptn, boundary_edges(ptn, central_verts; dir=:in))
  env_tensors = [mts[e] for e in pedges]
  env_tensors = reduce(vcat, env_tensors; init=ITensor[])
  central_tensors = ITensor[
    (unpartitioned_graph(ptn))[v] for v in setdiff(central_verts, verts)
  ]

  return vcat(env_tensors, central_tensors)
end

function environment_tensors(
  ptn::PartitionedGraph, mts, partition_verts::Vector{<:PartitionVertex}
)
  return environment_tensors(ptn, mts, vertices(ptn, partition_verts))
end
