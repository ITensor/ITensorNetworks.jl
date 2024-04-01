using NamedGraphs: vertex_to_parent_vertex
using ITensors: ITensor, scalar
using ITensors.ContractionSequenceOptimization: deepmap
using ITensors.NDTensors: NDTensors, Algorithm, @Algorithm_str, contract
using LinearAlgebra: normalize!

function NDTensors.contract(tn::AbstractITensorNetwork; alg::String="exact", kwargs...)
  return contract(Algorithm(alg), tn; kwargs...)
end

function NDTensors.contract(
  alg::Algorithm"exact", tn::AbstractITensorNetwork; sequence=vertices(tn), kwargs...
)
  sequence_linear_index = deepmap(v -> vertex_to_parent_vertex(tn, v), sequence)
  return contract(Vector{ITensor}(tn); sequence=sequence_linear_index, kwargs...)
end

function NDTensors.contract(alg::Algorithm"exact", tensors::Vector{ITensor}; kwargs...)
  return contract(tensors; kwargs...)
end

function NDTensors.contract(
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

function ITensors.scalar(
  alg::Algorithm, tn::Union{AbstractITensorNetwork,Vector{ITensor}}; kwargs...
)
  return contract(alg, tn; kwargs...)[]
end

function ITensors.scalar(
  tn::Union{AbstractITensorNetwork,Vector{ITensor}}; alg="exact", kwargs...
)
  return scalar(Algorithm(alg), tn; kwargs...)
end

function logscalar(
  tn::Union{AbstractITensorNetwork,Vector{ITensor}}; alg::String="exact", kwargs...
)
  return logscalar(Algorithm(alg), tn; kwargs...)
end

function logscalar(
  alg::Algorithm"exact", tn::Union{AbstractITensorNetwork,Vector{ITensor}}; kwargs...
)
  s = scalar(alg, tn; kwargs...)
  if s ≈ 0
    tol = 1e-16
    return -Inf
  elseif isa(s, AbstractFloat) && s >= 0
    return log(s)
  else
    return complex(s)
  end
end

function logscalar(
  alg::Algorithm,
  tn::AbstractITensorNetwork;
  (cache!)=nothing,
  cache_construction_kwargs=default_cache_construction_kwargs(alg, tn),
  update_cache=isnothing(cache!),
  cache_update_kwargs=default_cache_update_kwargs(cache!),
)
  if isnothing(cache!)
    cache! = Ref(cache(alg, tn; cache_construction_kwargs))
  end

  if update_cache
    cache![] = update(cache![]; cache_update_kwargs...)
  end

  numerator_terms, denominator_terms = scalar_factors(cache![])
  terms = vcat(numerator_terms, denominator_terms)
  if any(≈(0), terms)
    return -Inf
  elseif all(t -> isa(t, AbstractFloat), terms) && all(>=(0), terms)
    return sum(log.(numerator_terms)) - sum(log.((denominator_terms)))
  else
    return sum(log.(complex.(numerator_terms))) - sum(log.(complex.((denominator_terms))))
  end
end

function ITensors.scalar(alg::Algorithm"bp", tn::AbstractITensorNetwork; kwargs...)
  return exp(logscalar(alg, tn; kwargs...))
end
