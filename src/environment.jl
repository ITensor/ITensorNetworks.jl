default_environment_algorithm() = "exact"

function environment(
  ψ::AbstractITensorNetwork,
  vertices::Vector;
  alg=default_environment_algorithm(),
  kwargs...,
)
  return environment(Algorithm(alg), ψ, vertices; kwargs...)
end

function environment(
  ::Algorithm"exact",
  ψ::AbstractITensorNetwork,
  verts::Vector;
  contraction_sequence_alg="optimal",
  kwargs...,
)
  ψ_reduced = Vector{ITensor}(subgraph(ψ, setdiff(vertices(ψ), verts)))
  sequence = contraction_sequence(ψ_reduced; alg=contraction_sequence_alg)
  return ITensor[contract(ψ_reduced; sequence, kwargs...)]
end

function environment(
  ::Algorithm"bp",
  ψ::AbstractITensorNetwork,
  vertices::Vector;
  (cache!)=nothing,
  partitioned_vertices=default_partitioned_vertices(ψ),
  update_cache=isnothing(cache!),
  cache_update_kwargs=default_cache_update_kwargs(cache!),
)
  if isnothing(cache!)
    cache! = Ref(BeliefPropagationCache(ψ, partitioned_vertices))
  end

  if update_cache
    cache![] = update(cache![]; cache_update_kwargs...)
  end

  return environment(cache![], vertices)
end
