function default_cache(ψ::ITensorNetwork)
  ψψ = norm_network(ψ)
  return BeliefPropagationCache(ψψ, group(v -> v[1], vertices(ψψ)))
end
default_cache_update_kwargs(cache) = (; maxiter=20, tol=1e-5)

"""initialize bond tensors of an ITN to identity matrices"""
function initialize_bond_tensors(ψ::ITensorNetwork; index_map=prime)
  bond_tensors = DataGraph{vertextype(ψ),Nothing,ITensor}(underlying_graph(ψ))

  for e in edges(ψ)
    index = commoninds(ψ[src(e)], ψ[dst(e)])
    bond_tensors[e] = denseblocks(delta(index, index_map(index)))
  end

  return bond_tensors
end

"""Use an ITensorNetwork ψ, its bond tensors and belief propagation cache to put ψ into the vidal gauge, return the bond tensors and ψ_vidal."""
function vidal_gauge(
  ψ::ITensorNetwork,
  bond_tensors::DataGraph,
  bp_cache::BeliefPropagationCache;
  message_cutoff=10 * eps(real(scalartype(ψ))),
  regularization=10 * eps(real(scalartype(ψ))),
  edges=NamedGraphs.edges(ψ),
  svd_kwargs...,
)
  ψ_vidal = copy(ψ)

  for e in edges
    vsrc, vdst = src(e), dst(e)
    ψvsrc, ψvdst = ψ_vidal[vsrc], ψ_vidal[vdst]

    pe = partitionedge(bp_cache, (vsrc, 1) => (vdst, 1))
    edge_ind = commoninds(ψvsrc, ψvdst)
    edge_ind_sim = sim(edge_ind)

    X_D, X_U = eigen(only(message(bp_cache, pe)); ishermitian=true, cutoff=message_cutoff)
    Y_D, Y_U = eigen(
      only(message(bp_cache, reverse(pe))); ishermitian=true, cutoff=message_cutoff
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

function vidal_gauge(
  ψ::ITensorNetwork,
  bond_tensors;
  (cache!)=nothing,
  cache_update_kwargs=default_cache_update_kwargs(cache!),
  kwargs...,
)
  if isnothing(cache!)
    cache! = Ref(default_cache(ψ))
  end

  cache![] = update(cache![]; cache_update_kwargs...)

  return vidal_gauge(ψ, bond_tensors, cache![]; kwargs...)
end

"""Put an ITensorNetwork into the vidal gauge (by computing the message tensors), return the network and the bond tensors."""
function vidal_gauge(ψ::ITensorNetwork; kwargs...)
  bond_tensors = initialize_bond_tensors(ψ)
  return vidal_gauge(ψ, bond_tensors; kwargs...)
end

"""Transform from an ITensor in the Vidal Gauge (bond tensors) to the Symmetric Gauge (partitionedgraph, message tensors)"""
function symmetric_gauge(ψ::ITensorNetwork, bond_tensors::DataGraph)
  ψsymm = copy(ψ)
  ψψsymm = norm_network(ψsymm)

  bp_cache = BeliefPropagationCache(ψψsymm, group(v -> v[1], vertices(ψψsymm)))
  mts = messages(bp_cache)

  for e in edges(ψsymm)
    vsrc, vdst = src(e), dst(e)
    pe = partitionedge(bp_cache, (vsrc, 1) => (vdst, 1))
    root_S = sqrt_diag(bond_tensors[e])
    setindex_preserve_graph!(ψsymm, noprime(root_S * ψsymm[vsrc]), vsrc)
    setindex_preserve_graph!(ψsymm, noprime(root_S * ψsymm[vdst]), vdst)

    set!(mts, pe, copy(ITensor[dense(bond_tensors[e])]))
    set!(mts, reverse(pe), copy(ITensor[dense(bond_tensors[e])]))
  end

  ψψsymm = norm_network(ψsymm)
  pψψsymm = PartitionedGraph(ψψsymm, group(v -> v[1], vertices(ψψsymm)))

  return ψsymm, BeliefPropagationCache(pψψsymm, mts, default_message(bp_cache))
end

"""Put an ITensorNetwork into the symmetric gauge and also return the message tensors (which are the diagonal bond matrices from the Vidal Gauge)"""
function symmetric_gauge(ψ::ITensorNetwork; kwargs...)
  ψ_vidal, bond_tensors = vidal_gauge(ψ; kwargs...)

  return symmetric_gauge(ψ_vidal, bond_tensors)
end

"""Transform from the Symmetric Gauge (message tensors) to the Vidal Gauge (bond tensors)"""
function vidal_gauge(
  ψ::ITensorNetwork,
  bp_cache::BeliefPropagationCache;
  regularization=10 * eps(real(scalartype(ψ))),
)
  bond_tensors = DataGraph{vertextype(ψ),Nothing,ITensor}(underlying_graph(ψ))

  ψ_vidal = copy(ψ)

  for e in edges(ψ)
    vsrc, vdst = src(e), dst(e)
    pe = partitionedge(bp_cache, (vsrc, 1) => (vdst, 1))
    bond_tensors[e], bond_tensors[reverse(e)] = only(message(bp_cache, pe)),
    only(message(bp_cache, pe))
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
  isometries = Dict()

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

"""Function to measure the 'distance' of a state from the Vidal Gauge"""
function gauge_error(ψ::ITensorNetwork, bond_tensors::DataGraph)
  f = 0
  isometries = vidal_itn_isometries(ψ, bond_tensors)
  for e in keys(isometries)
    lhs = isometries[e]
    f += message_diff(ITensor[lhs], ITensor[dense(delta(inds(lhs)))])
  end

  return f / (length(keys(isometries)))
end

function gauge_error(ψ::ITensorNetwork, bp_cache::BeliefPropagationCache)
  Γ, Λ = vidal_gauge(ψ, bp_cache)
  return gauge_error(Γ, Λ)
end
