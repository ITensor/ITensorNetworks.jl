"""Put an ITensorNetwork into the symmetric gauge and also return the message tensors (which are the diagonal bond matrices)"""
function symmetric_gauge(ψ::ITensorNetwork; eigen_message_tensor_cutoff = 10*eps(real(scalartype(ψ))), regularization = 10*eps(real(scalartype(ψ))))
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])
  vertex_groups = nested_graph_leaf_vertices(partition(ψψ, group(v -> v[1], vertices(ψψ))))
  mts = compute_message_tensors(ψψ; vertex_groups)

  ψsymm = copy(ψ)
  symm_mts = copy(mts)

  for e in edges(ψ)
    vsrc, vdst = src(e), dst(e)

    s1, s2 = find_subgraph((vsrc, 1), mts), find_subgraph((vdst, 1), mts)
    edge_ind = commoninds(mts[s1 => s2], ψsymm[vsrc])
    edge_ind_sim = sim(edge_ind)

    X_D, X_U = eigen(mts[s1 => s2]; ishermitian=true, cutoff = eigen_message_tensor_cutoff)
    Y_D, Y_U = eigen(mts[s2 => s1]; ishermitian=true, cutoff = eigen_message_tensor_cutoff)

    X_D, Y_D = map_diag(x->x+regularization, X_D), map_diag(x->x+regularization, Y_D)

    rootX_D, rootY_D = sqrt_diag(X_D), sqrt_diag(Y_D)
    inv_rootX_D, inv_rootY_D = invsqrt_diag(X_D), invsqrt_diag(Y_D)
    rootX = X_U * rootX_D * prime(dag(X_U))
    rootY = Y_U * rootY_D * prime(dag(Y_U))
    inv_rootX = X_U * inv_rootX_D * prime(dag(X_U))
    inv_rootY = Y_U * inv_rootY_D * prime(dag(Y_U))

    ψsymm[vsrc] = noprime(ψsymm[vsrc] * inv_rootX)
    ψsymm[vdst] = noprime(ψsymm[vdst] * inv_rootY)

    Ce = rootX * replaceinds(rootY, edge_ind, edge_ind_sim)

    U, S, V = svd(Ce, edge_ind)
    rootS = sqrt_diag(S)

    ψsymm[vsrc] = replaceinds(ψsymm[vsrc] * U * rootS, commoninds(S, V), edge_ind)
    ψsymm[vdst] = replaceinds(ψsymm[vdst], edge_ind, edge_ind_sim)
    ψsymm[vdst] = replaceinds(ψsymm[vdst] * dag(rootS * V), commoninds(U, S), edge_ind)

    S = replaceinds(S, [commoninds(S, U)..., commoninds(S, V)...] => [edge_ind..., prime(edge_ind)...])
    symm_mts[s1 => s2], symm_mts[s2 => s1] = S, S

  end

  return ψsymm, symm_mts
end
