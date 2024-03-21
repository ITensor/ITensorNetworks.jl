function contract(tn::AbstractITensorNetwork; alg::String="exact", kwargs...)
  return contract(Algorithm(alg), tn; kwargs...)
end

function contract(
  alg::Algorithm"exact", tn::AbstractITensorNetwork; sequence=vertices(tn), kwargs...
)
  sequence_linear_index = deepmap(v -> vertex_to_parent_vertex(tn, v), sequence)
  return contract(Vector{ITensor}(tn); sequence=sequence_linear_index, kwargs...)
end

function contract(
  alg::Union{Algorithm"density_matrix",Algorithm"ttn_svd"},
  tn::AbstractITensorNetwork;
  output_structure::Function=path_graph_structure,
  kwargs...,
)
  return approx_itensornetwork(alg, tn, output_structure; kwargs...)
end

function contract_density_matrix(
  contract_list::Vector{ITensor}; normalize=true, contractor_kwargs...
)
  tn, _ = contract(
    ITensorNetwork(contract_list); alg="density_matrix", contractor_kwargs...
  )
  out = Vector{ITensor}(tn)
  if normalize
    out .= normalize!.(copy.(out))
  end
  return out
end

function contract_exact(
  contract_list::Vector{ITensor};
  contraction_sequence_alg="optimal",
  normalize=true,
  contractor_kwargs...,
)
  seq = contraction_sequence(contract_list; alg=contraction_sequence_alg)
  out = ITensors.contract(contract_list; sequence=seq, contractor_kwargs...)
  if normalize
    normalize!(out)
  end
  return ITensor[out]
end

scalar(tn::Union{AbstractITensorNetwork, Vector{ITensor}}; alg = "exact", kwargs...) = scalar(Algorithm(alg), tn; kwargs...)
scalar(alg::Algorithm, tn::Union{AbstractITensorNetwork, Vector{ITensor}}; kwargs...) = contract(alg, tn; kwargs...)[]

function logscalar(tn::AbstractITensorNetwork; alg::String="exact", kwargs...)
  return logscalar(Algorithm(alg), tn; kwargs...)
end

function logscalar(alg::Algorithm"exact", tn::AbstractITensorNetwork; kwargs...)
  return log(scalar(alg, tn; kwargs...))
end

function logscalar(alg::Algorithm"bp", tn::AbstractITensorNetwork; (cache!)=nothing,
  partitioned_vertices=default_partitioned_vertices(tn),
  update_cache=isnothing(cache!),
  cache_update_kwargs=default_cache_update_kwargs(cache!),)

  if isnothing(cache!)
    cache! = Ref(BeliefPropagationCache(tn, partitioned_vertices))
  end

  if update_cache
    cache![] = update(cache![]; cache_update_kwargs...)
  end

  pg = partitioned_itensornetwork(bp_cache)

  log_numerator, log_denominator = 0, 0
  for pv in partitionvertices(pg)
      incoming_mts = incoming_messages(bp_cache, [pv])
      local_state = factor(bp_cache, pv)
      log_numerator += logscalar(vcat(incoming_mts, local_state); alg = "exact")
  end
  for pe in partitionedges(pg)
      log_denominator += logscalar(vcat(message(bp_cache, pe), message(bp_cache, reverse(pe))); alg = "exact")
  end

  return log_numerator - log_denominator
end