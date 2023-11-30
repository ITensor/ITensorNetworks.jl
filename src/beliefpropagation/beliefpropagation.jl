function message_tensors(
  tn::ITensorNetwork;
  nvertices_per_partition=nothing,
  npartitions=nothing,
  subgraph_vertices=nothing,
  kwargs...,
)
  return message_tensors(
    partition(tn; nvertices_per_partition, npartitions, subgraph_vertices); kwargs...
  )
end

function message_tensors_skeleton(subgraphs::DataGraph)
  mts = DataGraph{vertextype(subgraphs),vertex_data_type(subgraphs),ITensorNetwork}(
    directed_graph(underlying_graph(subgraphs))
  )
  for v in vertices(mts)
    mts[v] = subgraphs[v]
  end

  return mts
end

function message_tensors(
  subgraphs::DataGraph;
  itensor_constructor=inds_e -> ITensor[dense(delta(i)) for i in inds_e],
)
  mts = message_tensors_skeleton(subgraphs)
  for e in edges(subgraphs)
    inds_e = commoninds(subgraphs[src(e)], subgraphs[dst(e)])
    itensors = itensor_constructor(inds_e)
    mts[e] = ITensorNetwork(itensors)
    mts[reverse(e)] = dag(mts[e])
  end
  return mts
end

"""
DO a single update of a message tensor using the current subgraph and the incoming mts
"""
function update_message_tensor(
  tn::ITensorNetwork,
  subgraph_vertices::Vector,
  mts::Vector{ITensorNetwork};
  contract_kwargs=(; alg="density_matrix", output_structure=path_graph_structure, maxdim=1),
)
  mts_itensors = reduce(vcat, ITensor.(mts); init=ITensor[])

  contract_list = ITensor[mts_itensors; ITensor[tn[v] for v in subgraph_vertices]]
  tn = if isone(length(contract_list))
    copy(only(contract_list))
  else
    ITensorNetwork(contract_list)
  end

  if contract_kwargs.alg != "exact"
    contract_output = contract(tn; contract_kwargs...)
  else
    contract_output = contract(tn; sequence=contraction_sequence(tn; alg="optimal"))
  end

  itn = if typeof(contract_output) == ITensor
    ITensorNetwork(contract_output)
  else
    first(contract_output)
  end
  normalize!.(vertex_data(itn))

  return itn
end

function update_message_tensor(
  tn::ITensorNetwork, subgraph::ITensorNetwork, mts::Vector{ITensorNetwork}; kwargs...
)
  return update_message_tensor(tn, vertices(subgraph), mts; kwargs...)
end

"""
Do a sequential update of message tensors on `edges` for a given ITensornetwork and its partition into sub graphs
"""
function belief_propagation_iteration(
  tn::ITensorNetwork,
  mts::DataGraph,
  edges::Vector{<:AbstractEdge};
  contract_kwargs=(; alg="density_matrix", output_structure=path_graph_structure, maxdim=1),
  compute_norm=false,
)
  new_mts = copy(mts)
  c = 0
  for e in edges
    environment_tensornetworks = ITensorNetwork[
      new_mts[e_in] for
      e_in in setdiff(boundary_edges(new_mts, [src(e)]; dir=:in), [reverse(e)])
    ]
    new_mts[src(e) => dst(e)] = update_message_tensor(
      tn, new_mts[src(e)], environment_tensornetworks; contract_kwargs
    )

    if compute_norm
      LHS, RHS = ITensors.contract(ITensor(mts[src(e) => dst(e)])),
      ITensors.contract(ITensor(new_mts[src(e) => dst(e)]))
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
  tn::ITensorNetwork,
  mts::DataGraph,
  edge_groups::Vector{<:Vector{<:AbstractEdge}};
  contract_kwargs=(; alg="density_matrix", output_structure=path_graph_structure, maxdim=1),
  compute_norm=false,
)
  new_mts = copy(mts)
  c = 0
  for edges in edge_groups
    updated_mts, ct = belief_propagation_iteration(
      tn, mts, edges; contract_kwargs, compute_norm
    )
    for e in edges
      new_mts[e] = updated_mts[e]
    end
    c += ct
  end
  return new_mts, c / (length(edge_groups))
end

function belief_propagation_iteration(
  tn::ITensorNetwork,
  mts::DataGraph;
  contract_kwargs=(; alg="density_matrix", output_structure=path_graph_structure, maxdim=1),
  compute_norm=false,
  edges=edge_sequence(mts),
)
  return belief_propagation_iteration(tn, mts, edges; contract_kwargs, compute_norm)
end

function belief_propagation(
  tn::ITensorNetwork,
  mts::DataGraph;
  contract_kwargs=(; alg="density_matrix", output_structure=path_graph_structure, maxdim=1),
  niters=default_bp_niters(mts),
  target_precision=nothing,
  edges=edge_sequence(mts),
  verbose=false,
)
  compute_norm = !isnothing(target_precision)
  if isnothing(niters)
    error("You need to specify a number of iterations for BP!")
  end
  for i in 1:niters
    mts, c = belief_propagation_iteration(tn, mts, edges; contract_kwargs, compute_norm)
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
  tn::ITensorNetwork;
  contract_kwargs=(; alg="density_matrix", output_structure=path_graph_structure, maxdim=1),
  nvertices_per_partition=nothing,
  npartitions=nothing,
  subgraph_vertices=nothing,
  niters=default_bp_niters(mts),
  target_precision=nothing,
  verbose=false,
)
  mts = message_tensors(tn; nvertices_per_partition, npartitions, subgraph_vertices)
  return belief_propagation(tn, mts; contract_kwargs, niters, target_precision, verbose)
end

"""
Given a subet of vertices of a given Tensor Network and the Message Tensors for that network, return a Dictionary with the involved subgraphs as keys and the vector of tensors associated with that subgraph as values
Specifically, the contraction of the environment tensors and tn[vertices] will be a scalar.
"""
function get_environment(tn::ITensorNetwork, mts::DataGraph, verts::Vector; dir=:in)
  subgraphs = unique([find_subgraph(v, mts) for v in verts])

  if dir == :out
    return get_environment(tn, mts, setdiff(vertices(tn), verts))
  end

  env_tns = ITensorNetwork[mts[e] for e in boundary_edges(mts, subgraphs; dir=:in)]
  central_tn = ITensorNetwork(
    ITensor[tn[v] for v in setdiff(flatten([vertices(mts[s]) for s in subgraphs]), verts)]
  )
  return ITensorNetwork(vcat(env_tns, ITensorNetwork[central_tn]))
end

"""
Calculate the contraction of a tensor network centred on the vertices verts. Using message tensors.
Defaults to using tn[verts] as the local network but can be overriden
"""
function approx_network_region(
  tn::ITensorNetwork,
  mts::DataGraph,
  verts::Vector;
  verts_tn=ITensorNetwork([tn[v] for v in verts]),
)
  environment_tn = get_environment(tn, mts, verts)

  return environment_tn ⊗ verts_tn
end
