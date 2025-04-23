using LinearAlgebra

function rescale(tn::AbstractITensorNetwork; alg="exact", kwargs...)
  return rescale(Algorithm(alg), tn; kwargs...)
end

function rescale(alg::Algorithm"exact", tn::AbstractITensorNetwork; kwargs...)
  logn = logscalar(alg, tn; kwargs...)
  vs = collect(vertices(tn))
  c = inv(exp(logn / length(vs)))
  vertices_weights = Dictionary(vs, [c for v in vs])
  return scale(tn, vertices_weights)
end

function rescale(
  alg::Algorithm,
  tn::AbstractITensorNetwork,
  args...;
  (cache!)=nothing,
  cache_construction_kwargs=default_cache_construction_kwargs(alg, tn),
  update_cache=isnothing(cache!),
  cache_update_kwargs=default_cache_update_kwargs(alg),
  kwargs...,
)
  if isnothing(cache!)
    cache! = Ref(cache(alg, tn; cache_construction_kwargs...))
  end

  if update_cache
    cache![] = update(cache![]; cache_update_kwargs...)
  end

  cache![] = rescale(cache![], args...; kwargs...)

  return tensornetwork(cache![])
end

function LinearAlgebra.normalize(tn::AbstractITensorNetwork; alg="exact", kwargs...)
  return normalize(Algorithm(alg), tn; kwargs...)
end

function LinearAlgebra.normalize(
  alg::Algorithm"exact", tn::AbstractITensorNetwork; kwargs...
)
  logn = logscalar(alg, inner_network(tn, tn); kwargs...)
  vs = collect(vertices(tn))
  c = inv(exp(logn / (2*length(vs))))
  vertices_weights = Dictionary(vs, [c for v in vs])
  return scale(tn, vertices_weights)
end

function LinearAlgebra.normalize(
  alg::Algorithm,
  tn::AbstractITensorNetwork;
  (cache!)=nothing,
  cache_construction_function=tn ->
    cache(alg, tn; default_cache_construction_kwargs(alg, tn)...),
  update_cache=isnothing(cache!),
  cache_update_kwargs=default_cache_update_kwargs(alg),
  cache_construction_kwargs=(;),
)
  norm_tn = inner_network(tn, tn)
  if isnothing(cache!)
    cache! = Ref(cache(alg, norm_tn; cache_construction_kwargs...))
  end

  vs = collect(vertices(tn))
  verts_to_rescale = vcat(
    [ket_vertex(norm_tn, v) for v in vs], [bra_vertex(norm_tn, v) for v in vs]
  )
  norm_tn = rescale(
    alg, norm_tn; verts_to_rescale, cache!, update_cache, cache_update_kwargs
  )

  return ket_network(norm_tn)
end
