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
  (bp_cache!)=nothing,
  update_bp_cache=isnothing(bp_cache!),
  bp_cache_update_kwargs=default_cache_update_kwargs(bp_cache!),
)
  if isnothing(bp_cache!)
    bp_cache! = Ref(BeliefPropagationCache(ψ))
  end

  if update_bp_cache
    bp_cache![] = update(bp_cache![]; bp_cache_update_kwargs...)
  end

  return incoming_messages(bp_cache![], setdiff(vertices(ψ), verts))
end
