"""initialize bond tensors of an ITN to identity matrices"""
function initialize_bond_tensors(ψ::ITensorNetwork; index_map=prime)
  bond_tensors = DataGraph{vertextype(ψ),ITensor,ITensor}(underlying_graph(ψ))

  for e in edges(ψ)
    index = commoninds(ψ[src(e)], ψ[dst(e)])
    bond_tensors[e] = dense(delta(index, index_map(index)))
  end

  return bond_tensors
end

"""Use an ITensorNetwork ψ, its bond tensors and gauging mts to put ψ into the vidal gauge, return the bond tensors and ψ_vidal."""
function vidal_gauge(
  ψ::ITensorNetwork,
  pψψ::PartitionedGraph,
  mts,
  bond_tensors::DataGraph;
  eigen_message_tensor_cutoff=10 * eps(real(scalartype(ψ))),
  regularization=10 * eps(real(scalartype(ψ))),
  edges=NamedGraphs.edges(ψ),
  svd_kwargs...,
)
  ψ_vidal = copy(ψ)

  for e in edges
    vsrc, vdst = src(e), dst(e)
    ψvsrc, ψvdst = ψ_vidal[vsrc], ψ_vidal[vdst]

    pe = partitionedge(pψψ, NamedEdge((vsrc, 1) => (vdst, 1)))
    edge_ind = commoninds(ψvsrc, ψvdst)
    edge_ind_sim = sim(edge_ind)

    X_D, X_U = eigen(only(mts[pe]); ishermitian=true, cutoff=eigen_message_tensor_cutoff)
    Y_D, Y_U = eigen(
      only(mts[reverse(pe)]); ishermitian=true, cutoff=eigen_message_tensor_cutoff
    )
    X_D, Y_D = map_diag(x -> x + regularization, X_D),
    map_diag(x -> x + regularization, Y_D)

    rootX_D, rootY_D = sqrt_diag(X_D), sqrt_diag(Y_D)
    inv_rootX_D, inv_rootY_D = invsqrt_diag(X_D), invsqrt_diag(Y_D)
    rootX = X_U * rootX_D * prime(dag(X_U))
    rootY = Y_U * rootY_D * prime(dag(Y_U))
    inv_rootX = X_U * inv_rootX_D * prime(dag(X_U))
    inv_rootY = Y_U * inv_rootY_D * prime(dag(Y_U))

    ψvsrc, ψvdst = noprime(ψvsrc * inv_rootX), noprime(ψvdst * inv_rootY)

    Ce = rootX * prime(bond_tensors[e])
    replaceinds!(Ce, edge_ind'', edge_ind')
    Ce = Ce * replaceinds(rootY, edge_ind, edge_ind_sim)

    U, S, V = svd(Ce, edge_ind; svd_kwargs...)

    new_edge_ind = Index[Index(dim(commoninds(S, U)), tags(first(edge_ind)))]

    ψvsrc = replaceinds(ψvsrc * U, commoninds(S, U), new_edge_ind)
    ψvdst = replaceinds(ψvdst, edge_ind, edge_ind_sim)
    ψvdst = replaceinds(ψvdst * V, commoninds(V, S), new_edge_ind)

    setindex_preserve_graph!(ψ_vidal, ψvsrc, vsrc)
    setindex_preserve_graph!(ψ_vidal, ψvdst, vdst)

    S = replaceinds(
      S,
      [commoninds(S, U)..., commoninds(S, V)...] =>
        [new_edge_ind..., prime(new_edge_ind)...],
    )
    bond_tensors[e] = S
  end

  return ψ_vidal, bond_tensors
end

"""Use an ITensorNetwork ψ in the symmetric gauge and its mts to put ψ into the vidal gauge. Return the bond tensors and ψ_vidal."""
function vidal_gauge(
  ψ::ITensorNetwork,
  pψψ::PartitionedGraph,
  mts;
  eigen_message_tensor_cutoff=10 * eps(real(scalartype(ψ))),
  regularization=10 * eps(real(scalartype(ψ))),
  edges=NamedGraphs.edges(ψ),
  svd_kwargs...,
)
  bond_tensors = initialize_bond_tensors(ψ)
  return vidal_gauge(
    ψ,
    pψψ,
    mts,
    bond_tensors;
    eigen_message_tensor_cutoff,
    regularization,
    edges,
    svd_kwargs...,
  )
end

"""Put an ITensorNetwork into the vidal gauge (by computing the message tensors), return the network and the bond tensors. Will also return the mts that were constructed"""
function vidal_gauge(
  ψ::ITensorNetwork;
  eigen_message_tensor_cutoff=10 * eps(real(scalartype(ψ))),
  regularization=10 * eps(real(scalartype(ψ))),
  niters=30,
  target_canonicalness::Union{Nothing,Float64}=nothing,
  verbose=false,
  svd_kwargs...,
)
  ψψ = norm_network(ψ)
  pψψ = PartitionedGraph(ψψ, collect(values(group(v -> v[1], vertices(ψψ)))))
  mts = message_tensors(pψψ)

  mts = belief_propagation(
    pψψ,
    mts;
    contract_kwargs=(; alg="exact"),
    niters,
    target_precision=target_canonicalness,
    verbose,
  )
  return vidal_gauge(
    ψ, pψψ, mts; eigen_message_tensor_cutoff, regularization, svd_kwargs...
  )
end

"""Transform from an ITensor in the Vidal Gauge (bond tensors) to the Symmetric Gauge (partitionedgraph, message tensors)"""
function vidal_to_symmetric_gauge(ψ::ITensorNetwork, bond_tensors::DataGraph)
  ψsymm = copy(ψ)
  ψψsymm = norm_network(ψsymm)
  pψψsymm = PartitionedGraph(ψψsymm, collect(values(group(v -> v[1], vertices(ψψsymm)))))
  ψsymm_mts = message_tensors(pψψsymm)

  for e in edges(ψsymm)
    vsrc, vdst = src(e), dst(e)
    pe = partitionedge(pψψsymm, NamedEdge((vsrc, 1) => (vdst, 1)))
    root_S = sqrt_diag(bond_tensors[e])
    setindex_preserve_graph!(ψsymm, noprime(root_S * ψsymm[vsrc]), vsrc)
    setindex_preserve_graph!(ψsymm, noprime(root_S * ψsymm[vdst]), vdst)

    ψsymm_mts[pe], ψsymm_mts[reverse(pe)] = copy(ITensor[dense(bond_tensors[e])]),
    copy(ITensor[dense(bond_tensors[e])])
  end

  ψψsymm = norm_network(ψsymm)
  pψψsymm = PartitionedGraph(ψψsymm, collect(values(group(v -> v[1], vertices(ψψsymm)))))

  return ψsymm, pψψsymm, ψsymm_mts
end

"""Put an ITensorNetwork into the symmetric gauge and also return the message tensors (which are the diagonal bond matrices from the Vidal Gauge)"""
function symmetric_gauge(
  ψ::ITensorNetwork;
  eigen_message_tensor_cutoff=10 * eps(real(scalartype(ψ))),
  regularization=10 * eps(real(scalartype(ψ))),
  niters=30,
  target_canonicalness::Union{Nothing,Float64}=nothing,
  svd_kwargs...,
)
  ψ_vidal, bond_tensors = vidal_gauge(
    ψ;
    eigen_message_tensor_cutoff,
    regularization,
    niters,
    target_canonicalness,
    svd_kwargs...,
  )

  return vidal_to_symmetric_gauge(ψ_vidal, bond_tensors)
end

"""Transform from the Symmetric Gauge (message tensors) to the Vidal Gauge (bond tensors)"""
function symmetric_to_vidal_gauge(
  ψ::ITensorNetwork,
  pψψ::PartitionedGraph,
  mts;
  regularization=10 * eps(real(scalartype(ψ))),
)
  bond_tensors = DataGraph{vertextype(ψ),ITensor,ITensor}(underlying_graph(ψ))

  ψ_vidal = copy(ψ)

  for e in edges(ψ)
    vsrc, vdst = src(e), dst(e)
    pe = partitionedge(pψψ, NamedEdge((vsrc, 1) => (vdst, 1)))
    bond_tensors[e] = only(mts[pe])
    invroot_S = invsqrt_diag(map_diag(x -> x + regularization, bond_tensors[e]))
    setindex_preserve_graph!(ψ_vidal, noprime(invroot_S * ψ_vidal[vsrc]), vsrc)
    setindex_preserve_graph!(ψ_vidal, noprime(invroot_S * ψ_vidal[vdst]), vdst)
  end

  return ψ_vidal, bond_tensors
end

"""Function to measure the 'isometries' of a state in the Vidal Gauge"""
function vidal_itn_isometries(
  ψ::ITensorNetwork,
  bond_tensors::DataGraph;
  edges=vcat(NamedGraphs.edges(ψ), reverse.(NamedGraphs.edges(ψ))),
)
  isometries = DataGraph{vertextype(ψ),ITensor,ITensor}(directed_graph(underlying_graph(ψ)))

  for e in edges
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
function symmetric_itn_canonicalness(ψ::ITensorNetwork, pψψ::PartitionedGraph, mts)
  ψ_vidal, bond_tensors = symmetric_to_vidal_gauge(ψ, pψψ, mts)

  return vidal_itn_canonicalness(ψ_vidal, bond_tensors)
end
