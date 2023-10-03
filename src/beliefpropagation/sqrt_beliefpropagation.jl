# using ITensors: scalartype
# using ITensorNetworks: find_subgraph, map_diag, sqrt_diag, boundary_edges

function sqrt_belief_propagation(
  tn::ITensorNetwork,
  mts::DataGraph;
  niters=20,
  update_order::String="parallel",
  # target_precision::Union{Float64,Nothing}=nothing,
)
  # compute_norm = target_precision == nothing ? false : true
  sqrt_mts = sqrt_message_tensors(tn, mts)
  for i in 1:niters
    sqrt_mts, c = sqrt_belief_propagation_iteration(tn, sqrt_mts; update_order) #; compute_norm)
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

function sqrt_belief_propagation_iteration(
  tn::ITensorNetwork,
  sqrt_mts::DataGraph;
  update_order::String="parallel",
  edges=Graphs.edges(sqrt_mts),

  # compute_norm=false,
)
  new_sqrt_mts = copy(sqrt_mts)
  if update_order != "parallel" && update_order != "sequential"
    error(
      "Specified update order is not currently implemented. Choose parallel or sequential."
    )
  end
  incoming_mts = update_order == "parallel" ? mts : new_mts
  c = 0.0
  for e in edges
    environment_tensornetworks = ITensorNetwork[
      sqrt_mts[e_in] for
      e_in in setdiff(boundary_edges(sqrt_mts, [src(e)]; dir=:in), [reverse(e)])
    ]

    new_sqrt_mts[src(e) => dst(e)] = update_sqrt_message_tensor(
      tn, sqrt_mts[src(e)], environment_tensornetworks;
    )

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

function update_sqrt_message_tensor(
  tn::ITensorNetwork, subgraph_vertices::Vector, sqrt_mts::Vector{ITensorNetwork};
)
  sqrt_mts_itensors = reduce(vcat, ITensor.(sqrt_mts); init=ITensor[])
  v = only(unique(first.(subgraph_vertices)))
  site_itensor = tn[v]
  contract_list = ITensor[sqrt_mts_itensors; site_itensor]
  contract_output = contract(
    contract_list; sequence=contraction_sequence(contract_list; alg="optimal")
  )
  left_inds = [uniqueinds(contract_output, site_itensor); siteinds(tn, v)]
  Q, R = qr(contract_output, left_inds)
  normalize!(R)
  return ITensorNetwork(R)
end

function update_sqrt_message_tensor(
  tn::ITensorNetwork, subgraph::ITensorNetwork, mts::Vector{ITensorNetwork}; kwargs...
)
  return update_sqrt_message_tensor(tn, vertices(subgraph), mts; kwargs...)
end

function sqrt_message_tensors(
  ψ::ITensorNetwork,
  mts::DataGraph;
  eigen_message_tensor_cutoff=10 * eps(real(scalartype(ψ))),
  regularization=10 * eps(real(scalartype(ψ))),
)
  sqrt_mts = copy(mts)
  for e in edges(ψ)
    vsrc, vdst = src(e), dst(e)
    ψvsrc, ψvdst = ψ[vsrc], ψ[vdst]

    s1, s2 = find_subgraph((vsrc, 1), mts), find_subgraph((vdst, 1), mts)
    edge_ind = commoninds(ψ[vsrc], ψ[vdst])
    edge_ind_sim = sim(edge_ind)

    X_D, X_U = eigen(
      contract(ITensor(mts[s1 => s2])); ishermitian=true, cutoff=eigen_message_tensor_cutoff
    )
    Y_D, Y_U = eigen(
      contract(ITensor(mts[s2 => s1])); ishermitian=true, cutoff=eigen_message_tensor_cutoff
    )
    X_D, Y_D = map_diag(x -> x + regularization, X_D),
    map_diag(x -> x + regularization, Y_D)

    rootX_D, rootY_D = sqrt_diag(X_D), sqrt_diag(Y_D)
    rootX = X_U * rootX_D * prime(dag(X_U))
    rootY = Y_U * rootY_D * prime(dag(Y_U))

    sqrt_mts[s1 => s2] = ITensorNetwork(rootX)
    sqrt_mts[s2 => s1] = ITensorNetwork(rootY)
  end
  return sqrt_mts
end

function sqr_message_tensors(sqrt_mts::DataGraph)
  mts = copy(sqrt_mts)
  for e in edges(sqrt_mts)
    sqrt_mt_tn = sqrt_mts[e]
    sqrt_mt = sqrt_mt_tn[only(vertices(sqrt_mt_tn))]
    sqrt_mt_rev_tn = sqrt_mts[reverse(e)]
    sqrt_mt_rev = sqrt_mt_rev_tn[only(vertices(sqrt_mt_rev_tn))]
    l = commoninds(sqrt_mt, sqrt_mt_rev)
    mt = dag(prime(sqrt_mt, l)) * sqrt_mt
    normalize!(mt)
    mt_rev = dag(prime(sqrt_mt_rev, l)) * sqrt_mt_rev
    normalize!(mt_rev)
    mts[e] = ITensorNetwork(mt)
    mts[reverse(e)] = ITensorNetwork(mt_rev)
  end
  return mts
end
