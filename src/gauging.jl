"""Use an ITensorNetwork ψ and its mts to put ψ into the vidal gauge, return the bond tensors and ψ_vidal."""
function vidal_gauge(
  ψ::ITensorNetwork, mts::DataGraph;
  eigen_message_tensor_cutoff=10 * eps(real(scalartype(ψ))),
  regularization=10 * eps(real(scalartype(ψ))),
  svd_kwargs...
)

  bond_tensors = DataGraph{vertextype(ψ),ITensor,ITensor}(underlying_graph(ψ))
  ψ_vidal = copy(ψ)

  for e in edges(ψ_vidal)
    vsrc, vdst = src(e), dst(e)

    s1, s2 = find_subgraph((vsrc, 1), mts), find_subgraph((vdst, 1), mts)
    edge_ind = commoninds(ψ_vidal[vsrc], ψ_vidal[vdst])
    edge_ind_sim = sim(edge_ind)

    X_D, X_U = eigen(ITensor(mts[s1 => s2]); ishermitian=true, cutoff=eigen_message_tensor_cutoff)
    Y_D, Y_U = eigen(ITensor(mts[s2 => s1]); ishermitian=true, cutoff=eigen_message_tensor_cutoff)
    X_D, Y_D = map_diag(x -> x + regularization, X_D), map_diag(x -> x + regularization, Y_D)

    rootX_D, rootY_D = sqrt_diag(X_D), sqrt_diag(Y_D)
    inv_rootX_D, inv_rootY_D = invsqrt_diag(X_D), invsqrt_diag(Y_D)
    rootX = X_U * rootX_D * prime(dag(X_U))
    rootY = Y_U * rootY_D * prime(dag(Y_U))
    inv_rootX = X_U * inv_rootX_D * prime(dag(X_U))
    inv_rootY = Y_U * inv_rootY_D * prime(dag(Y_U))

    ψ_vidal[vsrc] = noprime(ψ_vidal[vsrc] * inv_rootX)
    ψ_vidal[vdst] = noprime(ψ_vidal[vdst] * inv_rootY)

    Ce = rootX * replaceinds(rootY, edge_ind, edge_ind_sim)

    U, S, V = svd(Ce, edge_ind; svd_kwargs...)

    new_edge_ind = Index[Index(dim(commoninds(S, U)), tags(edge_ind[1]))]

    ψ_vidal[vsrc] = replaceinds(ψ_vidal[vsrc] * U, commoninds(S, U), new_edge_ind)
    ψ_vidal[vdst] = replaceinds(ψ_vidal[vdst], edge_ind, edge_ind_sim)
    ψ_vidal[vdst] = replaceinds(ψ_vidal[vdst] * V, commoninds(V, S), new_edge_ind)

    S = replaceinds(
      S, [commoninds(S, U)..., commoninds(S, V)...] => [new_edge_ind..., prime(new_edge_ind)...]
    )
    bond_tensors[e] = S
  end

  return ψ_vidal, bond_tensors
end

"""Put an ITensorNetwork into the vidal gauge, return the network and the bond tensors. Will also return the mts that were constructed"""
function vidal_gauge(
  ψ::ITensorNetwork;
  eigen_message_tensor_cutoff=10 * eps(real(scalartype(ψ))),
  regularization=10 * eps(real(scalartype(ψ))),
  niters=30,
  target_canonicalness::Union{Nothing, Float64}=nothing,
  svd_kwargs...
)
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])
  Z = partition(ψψ; subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ)))))
  mts = message_tensors(Z)

  if target_canonicalness == nothing
    mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"), niters)
    return vidal_gauge(ψ, mts; eigen_message_tensor_cutoff, regularization, niters, svd_kwargs...)
  else
    mts = belief_propagation_iteration(ψψ, mts; contract_kwargs=(; alg="exact"))
    ψ_vidal, bond_tensors = vidal_gauge(ψ, mts; eigen_message_tensor_cutoff, regularization, niters, svd_kwargs...)
    canonicalness = vidal_itn_canonicalness(ψ_vidal, bond_tensors)

    iter = 1
    while canonicalness > target_canonicalness && iter < niters
      mts = belief_propagation_iteration(ψψ, mts; contract_kwargs=(; alg="exact"))
      ψ_vidal, bond_tensors = vidal_gauge(ψ, mts; eigen_message_tensor_cutoff, regularization, svd_kwargs...)
      canonicalness = vidal_itn_canonicalness(ψ_vidal, bond_tensors)
      iter += 1
    end

    return ψ_vidal, bond_tensors
  end
end

"""Transform from the Vidal Gauge to the Symmetric Gauge"""
function vidal_to_symmetric_gauge(ψ::ITensorNetwork, bond_tensors::DataGraph)

  ψsymm = copy(ψ)
  ψψsymm = ψsymm ⊗ prime(dag(ψsymm); sites=[])
  Z = partition(ψψsymm; subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψsymm)))))
  ψsymm_mts = message_tensors_skeleton(Z)

  for e in edges(ψsymm)
    s1, s2 = find_subgraph((src(e), 1), ψsymm_mts), find_subgraph((dst(e), 1), ψsymm_mts)
    root_S = sqrt_diag(bond_tensors[e])
    ψsymm[src(e)] = noprime(ψsymm[src(e)] * root_S)
    ψsymm[dst(e)] = noprime(ψsymm[dst(e)] * root_S)

    ψsymm_mts[s1 => s2], ψsymm_mts[s2 => s1] = ITensorNetwork(bond_tensors[e]), ITensorNetwork(bond_tensors[e])
  end

  return ψsymm, ψsymm_mts

end

"""Put an ITensorNetwork into the symmetric gauge and also return the message tensors (which are the diagonal bond matrices from the Vidal Gauge)"""
function symmetric_gauge(
  ψ::ITensorNetwork;
  eigen_message_tensor_cutoff=10 * eps(real(scalartype(ψ))),
  regularization=10 * eps(real(scalartype(ψ))),
  niters=30,
  target_canonicalness::Union{Nothing, Float64}=nothing,
  svd_kwargs...
)
  ψsymm, bond_tensors = vidal_gauge(ψ; eigen_message_tensor_cutoff, regularization, niters, target_canonicalness, svd_kwargs...)

  return vidal_to_symmetric_gauge(ψsymm, bond_tensors)
end

"""Transform from the Symmetric Gauge to the Vidal Gauge"""
function symmetric_to_vidal_gauge(ψ::ITensorNetwork, mts::DataGraph; regularization=10 * eps(real(scalartype(ψ))))
  bond_tensors = DataGraph{vertextype(ψ),ITensor,ITensor}(underlying_graph(ψ))

  ψ_vidal = copy(ψ)

  for e in edges(ψ)
    s1, s2 = find_subgraph((src(e), 1), mts), find_subgraph((dst(e), 1), mts)
    bond_tensors[e] = ITensor(mts[s1=>s2])
    invroot_S = invsqrt_diag(map_diag(x -> x + regularization, bond_tensors[e]))
    ψ_vidal[src(e)] = noprime(invroot_S*ψ_vidal[src(e)])
    ψ_vidal[dst(e)] = noprime(invroot_S*ψ_vidal[dst(e)])
  end

  return ψ_vidal, bond_tensors
end

"""Function to measure the 'isometries' of a state in the Vidal Gauge"""
function vidal_itn_isometries(ψ::ITensorNetwork, bond_tensors::DataGraph)

  isometries = DataGraph{vertextype(ψ),ITensor,ITensor}(directed_graph(underlying_graph(ψ)))

  for e in vcat(edges(ψ), reverse.(edges(ψ)))
    vsrc, vdst = src(e), dst(e)
    ψv = copy(ψ[vsrc])
    for vn in setdiff(neighbors(ψ, vsrc), [vdst])
      ψv = noprime(ψv * bond_tensors[vn => vsrc])
    end

    ψvdag = dag(ψv)
    replaceind!(ψvdag, commonind(ψv, ψ[vdst]), commonind(ψv, ψ[vdst])')
    isometries[e] = ψvdag * ψv
  end

  return isometries
end

"""Function to measure the 'canonicalness' of a state in the Vidal Gauge"""
function vidal_itn_canonicalness(ψ::ITensorNetwork, bond_tensors::DataGraph)
  f = 0

  isometries = vidal_itn_isometries(ψ, bond_tensors)

  for e in edges(isometries)

    LHS = isometries[e] / sum(diag(isometries[e]))
    id = dense(delta(inds(LHS)))
    id /= sum(diag(id))
    f += 0.5 * norm(id - LHS)

  end

  return f / (length(edges(isometries)))
end

"""Function to measure the 'canonicalness' of a state in the Symmetric Gauge"""
function symmetric_itn_canonicalness(ψ::ITensorNetwork, mts::DataGraph)
  
  ψ_vidal, bond_tensors = symmetric_to_vidal_gauge(ψ, mts)

  return vidal_itn_canonicalness(ψ_vidal, bond_tensors)

end