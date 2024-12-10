using LinearAlgebra

function rescale(tn::AbstractITensorNetwork; alg="exact", kwargs...)
  return rescale(Algorithm(alg), tn; kwargs...)
end

function rescale(
  alg::Algorithm"exact", tn::AbstractITensorNetwork, vs=collect(vertices(tn)); kwargs...
)
  logn = logscalar(alg, tn; kwargs...)
  c = 1.0 / (exp(logn / length(vs)))
  tn = copy(tn)
  for v in vs
    tn[v] *= c
  end
  return tn
end

function rescale(
  alg::Algorithm,
  tn::AbstractITensorNetwork,
  vs=collect(vertices(tn));
  (cache!)=nothing,
  cache_construction_kwargs=default_cache_construction_kwargs(alg, tn),
  update_cache=isnothing(cache!),
  cache_update_kwargs=default_cache_update_kwargs(cache!),
)
  if isnothing(cache!)
    cache! = Ref(cache(alg, tn; cache_construction_kwargs...))
  end

  if update_cache
    cache![] = update(cache![]; cache_update_kwargs...)
  end

  tn = copy(tn)
  cache![] = normalize_messages(cache![])
  vertices_states = Dictionary()
  for pv in partitionvertices(cache![])
    pv_vs = filter(v -> v ∈ vs, vertices(cache![], pv))

    isempty(pv_vs) && continue

    vn = region_scalar(cache![], pv)
    if isreal(vn) && vn < 0
      tn[first(pv_vs)] *= -1
      vn = abs(vn)
    end

    vn = vn^(1 / length(pv_vs))
    for v in pv_vs
      tn[v] /= vn
      set!(vertices_states, v, tn[v])
    end
  end

  cache![] = update_factors(cache![], vertices_states)
  return tn
end

function LinearAlgebra.normalize(tn::AbstractITensorNetwork; alg="exact", kwargs...)
  return normalize(Algorithm(alg), tn; kwargs...)
end

function LinearAlgebra.normalize(
  alg::Algorithm"exact", tn::AbstractITensorNetwork; kwargs...
)
  norm_tn = QuadraticFormNetwork(tn)
  vs = filter(v -> v ∉ operator_vertices(norm_tn), collect(vertices(norm_tn)))
  return ket_network(rescale(alg, norm_tn, vs; kwargs...))
end

function LinearAlgebra.normalize(
  alg::Algorithm,
  tn::AbstractITensorNetwork;
  (cache!)=nothing,
  cache_construction_function=tn ->
    cache(alg, tn; default_cache_construction_kwargs(alg, tn)...),
  update_cache=isnothing(cache!),
  cache_update_kwargs=default_cache_update_kwargs(cache!),
)
  norm_tn = QuadraticFormNetwork(tn)
  if isnothing(cache!)
    cache! = Ref(cache_construction_function(norm_tn))
  end

  vs = filter(v -> v ∉ operator_vertices(norm_tn), collect(vertices(norm_tn)))
  norm_tn = rescale(alg, norm_tn, vs; cache!, update_cache, cache_update_kwargs)

  return ket_network(norm_tn)
end
