"""Modify the diagonal elements of an ITensor with a specified function"""
function modify_diagonal_els_itensor(A::ITensor, f::Function)
  out = copy(A)
  is = [[j for i in 1:length(inds(A))] for j in 1:mindim(out)]
  for i in is
    out[i...] = f(out[i...])
  end

  return out
end

"""Put an ITensorNetwork into the symmetric gauge and also return the message tensors (which are the diagonal bond matrices)"""
function symmetrise_itensornetwork(ψ::ITensorNetwork)
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])
  vertex_groups = nested_graph_leaf_vertices(partition(ψψ, group(v -> v[1], vertices(ψψ))))
  mts = compute_message_tensors(ψψ; vertex_groups=vertex_groups)

  ψsymm = copy(ψ)
  symm_mts = copy(mts)

  for e in edges(ψ)
    vsrc, vdst = src(e), dst(e)

    s1, s2 = find_subgraph((vsrc, 1), mts), find_subgraph((vdst, 1), mts)
    edge_ind = commoninds(mts[s1 => s2], ψsymm[vsrc])
    edge_ind_sim = sim(edge_ind)

    X_D, X_U = eigen(mts[s1 => s2]; ishermitian=true)
    Y_D, Y_U = eigen(mts[s2 => s1]; ishermitian=true)

    rootX_D, rootY_D = modify_diagonal_els_itensor(X_D, x -> x^0.5),
    modify_diagonal_els_itensor(Y_D, x -> x^0.5)
    inv_rootX_D, inv_rootY_D = modify_diagonal_els_itensor(X_D, x -> x^-0.5),
    modify_diagonal_els_itensor(Y_D, x -> x^-0.5)
    rootX = X_U * rootX_D * prime(dag(X_U))
    rootY = Y_U * rootY_D * prime(dag(Y_U))
    inv_rootX = X_U * inv_rootX_D * prime(dag(X_U))
    inv_rootY = Y_U * inv_rootY_D * prime(dag(Y_U))

    ψsymm[vsrc] = noprime(ψsymm[vsrc] * inv_rootX)
    ψsymm[vdst] = noprime(ψsymm[vdst] * inv_rootY)

    Ce = rootX * replaceinds(rootY, edge_ind, edge_ind_sim)

    U, S, V = svd(Ce, edge_ind)
    rootS = modify_diagonal_els_itensor(S, x -> x^0.5)

    ψsymm[vsrc] = replaceinds(ψsymm[vsrc] * U * rootS, commoninds(S, V), edge_ind)
    ψsymm[vdst] = replaceinds(ψsymm[vdst], edge_ind, edge_ind_sim)
    ψsymm[vdst] = replaceinds(ψsymm[vdst] * dag(rootS * V), commoninds(U, S), edge_ind)

    replaceinds!(S, inds(S), inds(mts[s1 => s2]))
    symm_mts[s1 => s2] = S
    symm_mts[s2 => s1] = S
  end

  return ψsymm, symm_mts
end
