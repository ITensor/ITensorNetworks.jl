# using ITensors: scalartype
# using ITensorNetworks: find_subgraph, map_diag, sqrt_diag, boundary_edges

function sqrt_belief_propagation_iteration(
  pψψ::PartitionedGraph, sqrt_mts, edges::Vector{<:PartitionEdge}
)
  new_sqrt_mts = copy(sqrt_mts)
  c = 0.0
  for e in edges
    new_sqrt_mts[e] = ITensor[update_sqrt_message_tensor(pψψ, e, new_sqrt_mts;)]

    # if compute_norm
    #   LHS, RHS = ITensors.contract(ITensor(sqrt_mts[src(e) => dst(e)])),
    #   ITensors.contract(ITensor(new_sqrt_mts[src(e) => dst(e)]))
    #   LHS /= sum(diag(LHS))
    #   RHS /= sum(diag(RHS))
    #   c += 0.5 * norm(LHS - RHS)
    # end
  end
  return new_sqrt_mts, c / (length(edges))
end

function sqrt_belief_propagation_iteration(
  pψψ::PartitionedGraph, sqrt_mts, edges::Vector{<:Vector{<:PartitionEdge}}
)
  new_sqrt_mts = copy(sqrt_mts)
  c = 0.0
  for e_group in edges
    updated_sqrt_mts, ct = sqrt_belief_propagation_iteration(pψψ, sqr_mts, e_group)
    for e in e_group
      new_sqrt_mts[e] = updated_sqrt_mts[e]
    end
    c += ct
  end
  return new_sqrt_mts, c / (length(edges))
end

function sqrt_belief_propagation_iteration(
  pψψ::PartitionedGraph, sqrt_mts; edges=edge_sequence(partitioned_graph(pψψ))
)
  return sqrt_belief_propagation_iteration(pψψ, sqrt_mts, edges)
end

function sqrt_belief_propagation(
  pψψ::PartitionedGraph,
  mts;
  niters=default_bp_niters(partitioned_graph(pψψ)),
  edges=PartitionEdge.(edge_sequence(partitioned_graph(pψψ))),
  # target_precision::Union{Float64,Nothing}=nothing,
)
  # compute_norm = target_precision == nothing ? false : true
  sqrt_mts = sqrt_message_tensors(pψψ, mts)
  if isnothing(niters)
    error("You need to specify a number of iterations for BP!")
  end
  for i in 1:niters
    sqrt_mts, c = sqrt_belief_propagation_iteration(pψψ, sqrt_mts, edges) #; compute_norm)
    # if compute_norm && c <= target_precision
    #   println(
    #     "Belief Propagation finished. Reached a canonicalness of " *
    #     string(c) *
    #     " after $i iterations. ",
    #   )
    #   break
    # end
  end
  return sqr_message_tensors(sqrt_mts)
end

function update_sqrt_message_tensor(
  pψψ::PartitionedGraph, edge::PartitionEdge, sqrt_mts;
)
  v = only(filter(v -> v[2] == 1, vertices(pψψ, src(edge))))
  site_itensor = unpartitioned_graph(pψψ)[v]
  incoming_messages = [
    sqrt_mts[PartitionEdge(e_in)] for e_in in setdiff(boundary_edges(partitioned_graph(pψψ), [NamedGraphs.parent(src(edge))]; dir=:in), [reverse(NamedGraphs.parent(edge))])
  ]
  incoming_messages = reduce(vcat, incoming_messages; init=ITensor[])
  contract_list = ITensor[incoming_messages; site_itensor]
  contract_output = contract(
    contract_list; sequence=contraction_sequence(contract_list; alg="optimal")
  )
  left_inds = [uniqueinds(contract_output, site_itensor); siteinds(unpartitioned_graph(pψψ), v)]
  Q, R = qr(contract_output, left_inds)
  normalize!(R)
  return R
end

function sqrt_message_tensors(
  pψψ::PartitionedGraph,
  mts;
  eigen_message_tensor_cutoff=10 * eps(real(scalartype(unpartitioned_graph(pψψ)))),
  regularization=10 * eps(real(scalartype(unpartitioned_graph(pψψ)))),
)
  sqrt_mts = copy(mts)
  for e in PartitionEdge.(edges(partitioned_graph(pψψ)))
    vsrc, vdst = filter(v -> v[2] == 1, vertices(pψψ, src(e))), filter(v -> v[2] == 1, vertices(pψψ, dst(e)))
    ψvsrc, ψvdst = unpartitioned_graph(pψψ)[only(vsrc)],unpartitioned_graph(pψψ)[only(vdst)]

    edge_ind = commoninds(ψvsrc, ψvdst)
    edge_ind_sim = sim(edge_ind)

    X_D, X_U = eigen(
      only(mts[e]); ishermitian=true, cutoff=eigen_message_tensor_cutoff
    )
    Y_D, Y_U = eigen(
      only(mts[PartitionEdge(reverse(NamedGraphs.parent(e)))]); ishermitian=true, cutoff=eigen_message_tensor_cutoff
    )
    X_D, Y_D = map_diag(x -> x + regularization, X_D),
    map_diag(x -> x + regularization, Y_D)

    rootX_D, rootY_D = sqrt_diag(X_D), sqrt_diag(Y_D)
    rootX = X_U * rootX_D * prime(dag(X_U))
    rootY = Y_U * rootY_D * prime(dag(Y_U))

    sqrt_mts[e] = ITensor[rootX]
    sqrt_mts[PartitionEdge(reverse(NamedGraphs.parent(e)))] = ITensor[rootY]
  end
  return sqrt_mts
end

function sqr_message_tensors(sqrt_mts)
  mts = copy(sqrt_mts)
  for e in keys(sqrt_mts)
    sqrt_mt = only(sqrt_mts[e])
    sqrt_mt_rev = only(sqrt_mts[PartitionEdge(reverse(NamedGraphs.parent(e)))])
    l = commoninds(sqrt_mt, sqrt_mt_rev)
    mt = dag(prime(sqrt_mt, l)) * sqrt_mt
    normalize!(mt)
    mt_rev = dag(prime(sqrt_mt_rev, l)) * sqrt_mt_rev
    normalize!(mt_rev)
    mts[e] = ITensor[mt]
    mts[PartitionEdge(reverse(NamedGraphs.parent(e)))] = ITensor[mt_rev]
  end
  return mts
end
