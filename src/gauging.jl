function default_bond_tensor_cache(ψ::ITensorNetwork)
  return DataGraph{vertextype(ψ),Nothing,ITensor}(underlying_graph(ψ))
end
default_bond_tensor(index) = denseblocks(delta(index, prime(index)))

struct VidalITensorNetwork{V,BTS} <: AbstractITensorNetwork{V}
  itensornetwork::ITensorNetwork{V}
  bond_tensors::BTS
end

itensornetwork(ψv::VidalITensorNetwork) = ψv.itensornetwork
bond_tensors(ψv::VidalITensorNetwork) = ψv.bond_tensors
bond_tensor(ψv::VidalITensorNetwork, e) = bond_tensors(ψv)[e]

data_graph_type(::Type{VidalITensorNetwork}) = data_graph_type(itensornetwork(ψv))
data_graph(ψv::VidalITensorNetwork) = data_graph(itensornetwork(ψv))
function copy(ψv::VidalITensorNetwork)
  return VidalITensorNetwork(copy(itensornetwork(ψv)), copy(bond_tensors(ψv)))
end

function VidalITensorNetwork(
  ψ::ITensorNetwork;
  bond_tensors_cache=default_bond_tensor_cache,
  bond_tensor=default_bond_tensor,
)
  bond_tensors = bond_tensors_cache(ψ)
  for e in edges(ψ)
    index = commoninds(ψ[src(e)], ψ[dst(e)])
    bond_tensors[e] = bond_tensor(index)
  end

  return VidalITensorNetwork(ψ, bond_tensors)
end

function default_norm_cache(ψ::ITensorNetwork)
  ψψ = norm_network(ψ)
  return BeliefPropagationCache(ψψ, group(v -> v[1], vertices(ψψ)))
end
default_cache_update_kwargs(cache) = (; maxiter=20, tol=1e-5)

"""Use an ITensorNetwork ψ, its bond tensors and belief propagation cache to put ψ into the vidal gauge, return the bond tensors and updated_ψ."""
function update(
  ψv::VidalITensorNetwork,
  bp_cache::BeliefPropagationCache;
  message_cutoff=10 * eps(real(scalartype(ψv))),
  regularization=10 * eps(real(scalartype(ψv))),
  edges=NamedGraphs.edges(ψv),
  svd_kwargs...,
)
  updated_ψ = copy(itensornetwork(ψv))
  updated_bond_tensors = copy(bond_tensors(ψv))

  for e in edges
    vsrc, vdst = src(e), dst(e)
    ψvsrc, ψvdst = updated_ψ[vsrc], updated_ψ[vdst]

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

    Ce = rootX * prime(bond_tensor(ψv, e))
    replaceinds!(Ce, edge_ind'', edge_ind')
    Ce = Ce * replaceinds(rootY, edge_ind, edge_ind_sim)

    U, S, V = svd(Ce, edge_ind; svd_kwargs...)

    new_edge_ind = Index[Index(dim(commoninds(S, U)), tags(first(edge_ind)))]

    ψvsrc = replaceinds(ψvsrc * U, commoninds(S, U), new_edge_ind)
    ψvdst = replaceinds(ψvdst, edge_ind, edge_ind_sim)
    ψvdst = replaceinds(ψvdst * V, commoninds(V, S), new_edge_ind)

    setindex_preserve_graph!(updated_ψ, ψvsrc, vsrc)
    setindex_preserve_graph!(updated_ψ, ψvdst, vdst)

    S = replaceinds(
      S,
      [commoninds(S, U)..., commoninds(S, V)...] =>
        [new_edge_ind..., prime(new_edge_ind)...],
    )
    updated_bond_tensors[e] = S
  end

  return VidalITensorNetwork(updated_ψ, updated_bond_tensors)
end

function update(
  ψv::VidalITensorNetwork;
  (cache!)=nothing,
  cache_update_kwargs=default_cache_update_kwargs(cache!),
  kwargs...,
)
  if isnothing(cache!)
    cache! = Ref(default_norm_cache(itensornetwork(ψv)))
  end

  cache![] = update(cache![]; cache_update_kwargs...)

  return update(ψv, cache![]; kwargs...)
end

"""Transform from an ITensor in the Vidal Gauge (bond tensors) to the Symmetric Gauge (partitionedgraph, message tensors)"""
function symmetric_gauge(ψv::VidalITensorNetwork)
  ψsymm = copy(itensornetwork(ψv))

  bp_cache = default_norm_cache(ψsymm)
  mts = messages(bp_cache)

  for e in edges(ψsymm)
    vsrc, vdst = src(e), dst(e)
    pe = partitionedge(bp_cache, (vsrc, 1) => (vdst, 1))
    root_S = sqrt_diag(bond_tensor(ψv, e))
    setindex_preserve_graph!(ψsymm, noprime(root_S * ψsymm[vsrc]), vsrc)
    setindex_preserve_graph!(ψsymm, noprime(root_S * ψsymm[vdst]), vdst)

    set!(mts, pe, copy(ITensor[dense(bond_tensor(ψv, e))]))
    set!(mts, reverse(pe), copy(ITensor[dense(bond_tensor(ψv, e))]))
  end

  bp_cache = default_norm_cache(ψsymm)

  return ψsymm, set_messages(bp_cache, mts)
end

"""Put an ITensorNetwork into the symmetric gauge and also return the message tensors (which are the diagonal bond matrices from the Vidal Gauge)"""
function symmetric_gauge(ψ::ITensorNetwork; kwargs...)
  ψv = VidalITensorNetwork(ψ)
  ψv = update(ψv; kwargs...)

  return symmetric_gauge(ψv)
end

"""Function to measure the 'isometries' of a state in the Vidal Gauge"""
function vidal_gauge_isometries(
  ψv::VidalITensorNetwork;
  edges=vcat(NamedGraphs.edges(ψv), reverse.(NamedGraphs.edges(ψv))),
)
  isometries = Dict()

  for e in edges
    vsrc, vdst = src(e), dst(e)
    ψ_vsrc = copy(ψv[vsrc])
    for vn in setdiff(neighbors(ψv, vsrc), [vdst])
      ψ_vsrc = noprime(ψ_vsrc * bond_tensor(ψv, vn => vsrc))
    end

    ψ_vsrcdag = dag(ψ_vsrc)
    replaceind!(ψ_vsrcdag, commonind(ψ_vsrc, ψv[vdst]), commonind(ψ_vsrc, ψv[vdst])')
    isometries[e] = ψ_vsrcdag * ψ_vsrc
  end

  return isometries
end

"""Function to measure the 'distance' of a state from the Vidal Gauge"""
function gauge_error(ψv::VidalITensorNetwork)
  f = 0
  isometries = vidal_gauge_isometries(ψv)
  for e in keys(isometries)
    lhs = isometries[e]
    f += message_diff(ITensor[lhs], ITensor[denseblocks(delta(inds(lhs)))])
  end

  return f / (length(keys(isometries)))
end
