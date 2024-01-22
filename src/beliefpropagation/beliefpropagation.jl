function message_tensors(
  ptn::PartitionedGraph; itensor_constructor=inds_e -> ITensor[dense(delta(inds_e))]
)
  mts = Dict()
  for e in partitionedges(ptn)
    src_e_itn = unpartitioned_graph(subgraph(ptn, [src(e)]))
    dst_e_itn = unpartitioned_graph(subgraph(ptn, [dst(e)]))
    inds_e = commoninds(src_e_itn, dst_e_itn)
    mts[e] = itensor_constructor(inds_e)
    mts[reverse(e)] = dag.(mts[e])
  end
  return mts
end

"""
Do a single update of a message tensor using the current subgraph and the incoming mts
"""
function update_message_tensor(
  ptn::PartitionedGraph,
  edge::PartitionEdge,
  mts;
  contract_kwargs=(; alg="density_matrix", output_structure=path_graph_structure, maxdim=1),
)
  pedges = setdiff(
    partitionedges(ptn, boundary_edges(ptn, vertices(ptn, src(edge)); dir=:in)),
    [reverse(edge)],
  )
  incoming_messages = [mts[e_in] for e_in in pedges]
  incoming_messages = reduce(vcat, incoming_messages; init=ITensor[])

  contract_list = ITensor[
    incoming_messages
    ITensor(unpartitioned_graph(subgraph(ptn, [src(edge)])))
  ]

  if contract_kwargs.alg != "exact"
    mt = first(contract(ITensorNetwork(contract_list); contract_kwargs...))
  else
    mt = contract(
      contract_list; sequence=contraction_sequence(contract_list; alg="optimal")
    )
  end

  mt = isa(mt, ITensor) ? ITensor[mt] : ITensor(mt)
  normalize!.(mt)

  return mt
end

"""
Do a sequential update of message tensors on `edges` for a given ITensornetwork and its partition into sub graphs
"""
function belief_propagation_iteration(
  ptn::PartitionedGraph,
  mts,
  edges::Vector{<:PartitionEdge};
  contract_kwargs=(; alg="density_matrix", output_structure=path_graph_structure, maxdim=1),
  compute_norm=false,
)
  new_mts = copy(mts)
  c = 0
  for e in edges
    new_mts[e] = update_message_tensor(ptn, e, new_mts; contract_kwargs)

    if compute_norm
      LHS, RHS = ITensors.contract(mts[e]), ITensors.contract(new_mts[e])
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
  contract_kwargs=(; alg="density_matrix", output_structure=path_graph_structure, maxdim=1),
  compute_norm=false,
)
  new_mts = copy(mts)
  c = 0
  for edges in edge_groups
    updated_mts, ct = belief_propagation_iteration(
      ptn, mts, edges; contract_kwargs, compute_norm
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
  contract_kwargs=(; alg="density_matrix", output_structure=path_graph_structure, maxdim=1),
  compute_norm=false,
  edges=PartitionEdge.(edge_sequence(partitioned_graph(ptn))),
)
  return belief_propagation_iteration(ptn, mts, edges; contract_kwargs, compute_norm)
end

function belief_propagation(
  ptn::PartitionedGraph,
  mts;
  contract_kwargs=(; alg="density_matrix", output_structure=path_graph_structure, maxdim=1),
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
    mts, c = belief_propagation_iteration(ptn, mts, edges; contract_kwargs, compute_norm)
    if compute_norm && c <= target_precision
      if verbose
        println("BP converged to desired precision after $i iterations.")
      end
      break
    end
  end
  return mts
end

function belief_propagation(
  ptn::PartitionedGraph;
  itensor_constructor=inds_e -> ITensor[dense(delta(inds_e))],
  kwargs...,
)
  mts = message_tensors(ptn; itensor_constructor)
  return belief_propagation(ptn, mts; kwargs...)
end
"""
Given a subet of vertices of a given Tensor Network and the Message Tensors for that network, return a Dictionary with the involved subgraphs as keys and the vector of tensors associated with that subgraph as values
Specifically, the contraction of the environment tensors and tn[vertices] will be a scalar.
"""
function get_environment(ptn::PartitionedGraph, mts, verts::Vector; dir=:in)
  partition_verts = partitionvertices(ptn, verts)
  central_verts = vertices(ptn, partition_verts)

  if dir == :out
    return get_environment(ptn, mts, setdiff(vertices(ptn), verts))
  end

  pedges = partitionedges(ptn, boundary_edges(ptn, central_verts; dir=:in))
  env_tensors = [mts[e] for e in pedges]
  env_tensors = reduce(vcat, env_tensors; init=ITensor[])
  central_tensors = ITensor[
    (unpartitioned_graph(ptn))[v] for v in setdiff(central_verts, verts)
  ]

  return vcat(env_tensors, central_tensors)
end

"""
Calculate the contraction of a tensor network centred on the vertices verts. Using message tensors.
Defaults to using tn[verts] as the local network but can be overriden
"""
function approx_network_region(
  ptn::PartitionedGraph,
  mts,
  verts::Vector;
  verts_tensors=ITensor[(unpartitioned_graph(ptn))[v] for v in verts],
)
  environment_tensors = get_environment(ptn, mts, verts)

  return vcat(environment_tensors, verts_tensors)
end
