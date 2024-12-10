using LinearAlgebra

function rescale(tn::AbstractITensorNetwork, c::Number, vs=collect(vertices(tn)))
  tn = copy(tn)
  for v in vs
    tn[v] *= c
  end
  return tn
end

function LinearAlgebra.normalize(tn::AbstractITensorNetwork; alg="exact", kwargs...)
  return normalize(Algorithm(alg), tn; kwargs...)
end

function LinearAlgebra.normalize(alg::Algorithm"exact", tn::AbstractITensorNetwork)
  norm_tn = QuadraticFormNetwork(tn)
  c = exp(logscalar(alg, norm_tn) / (2 * length(vertices(tn))))
  return rescale(tn, 1 / c)
end

function LinearAlgebra.normalize(
  alg::Algorithm"bp",
  tn::AbstractITensorNetwork;
  (cache!)=nothing,
  update_cache=isnothing(cache!),
  cache_update_kwargs=default_cache_update_kwargs(cache!),
)
  if isnothing(cache!)
    cache! = Ref(BeliefPropagationCache(QuadraticFormNetwork(tn)))
  end

  if update_cache
    cache![] = update(cache![]; cache_update_kwargs...)
  end

  tn = copy(tn)
  cache![] = normalize_messages(cache![])
  norm_tn = tensornetwork(cache![])

  vertices_states = Dictionary()
  for v in vertices(tn)
    v_ket, v_bra = ket_vertex(norm_tn, v), bra_vertex(norm_tn, v)
    pv = only(partitionvertices(cache![], [v_ket]))
    vn = region_scalar(cache![], pv)
    norm_tn = rescale(norm_tn, 1 / sqrt(vn), [v_ket, v_bra])
    set!(vertices_states, v_ket, norm_tn[v_ket])
    set!(vertices_states, v_bra, norm_tn[v_bra])
  end

  cache![] = update_factors(cache![], vertices_states)

  return ket_network(norm_tn)
end
