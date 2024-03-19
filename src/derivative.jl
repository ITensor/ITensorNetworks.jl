default_derivative_algorithm() = "exact"

function derivative(
  ψ::AbstractITensorNetwork, vertices::Vector; alg=default_derivative_algorithm(), kwargs...
)
  return derivative(Algorithm(alg), ψ, vertices; kwargs...)
end

function derivative(
  ::Algorithm"exact",
  ψ::AbstractITensorNetwork,
  vertices::Vector;
  contraction_sequence_alg="optimal",
  kwargs...,
)
  ψ_reduced = Vector{ITensor}(subgraph(ψ, vertices))
  sequence = contraction_sequence(ψ_reduced; alg=contraction_sequence_alg)
  return ITensor[contract(ψ_reduced; sequence, kwargs...)]
end

function derivative(
  ::Algorithm"bp",
  ψ::AbstractITensorNetwork,
  verts::Vector;
  (cache!)=nothing,
  update_cache=isnothing(cache!),
  cache_update_kwargs=default_cache_update_kwargs(cache!),
)
  if isnothing(cache!)
    cache! = Ref(BeliefPropagationCache(ψ))
  end

  if update_cache
    cache![] = update(cache![]; cache_update_kwargs...)
  end

  return incoming_messages(cache![], setdiff(vertices(ψ), verts))
end
