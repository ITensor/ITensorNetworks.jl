using ITensors: ITensor, scalar
using ITensors.ContractionSequenceOptimization: deepmap
using ITensors.NDTensors: NDTensors, Algorithm, @Algorithm_str, contract
using LinearAlgebra: normalize!
using NamedGraphs: NamedGraphs
using NamedGraphs.OrdinalIndexing: th

function NDTensors.contract(tn::AbstractITensorNetwork; alg="exact", kwargs...)
  return contract(Algorithm(alg), tn; kwargs...)
end

function NDTensors.contract(
  alg::Algorithm"exact",
  tn::AbstractITensorNetwork;
  contraction_sequence_kwargs=(;),
  sequence=contraction_sequence(tn; contraction_sequence_kwargs...),
  kwargs...,
)
  sequence_linear_index = deepmap(v -> NamedGraphs.vertex_positions(tn)[v], sequence)
  ts = map(v -> tn[v], (1:nv(tn))th)
  return contract(ts; sequence=sequence_linear_index, kwargs...)
end

function NDTensors.contract(
  alg::Union{Algorithm"density_matrix",Algorithm"ttn_svd"},
  tn::AbstractITensorNetwork;
  output_structure::Function=path_graph_structure,
  kwargs...,
)
  return contract_approx(alg, tn, output_structure; kwargs...)
end

function ITensors.scalar(alg::Algorithm"exact", tn::AbstractITensorNetwork; kwargs...)
  return contract(alg, tn; kwargs...)[]
end

function ITensors.scalar(tn::AbstractITensorNetwork; alg="exact", kwargs...)
  return scalar(Algorithm(alg), tn; kwargs...)
end

function logscalar(tn::AbstractITensorNetwork; alg="exact", kwargs...)
  return logscalar(Algorithm(alg), tn; kwargs...)
end

function logscalar(alg::Algorithm"exact", tn::AbstractITensorNetwork; kwargs...)
  s = scalar(alg, tn; kwargs...)
  s = real(s) < 0 ? complex(s) : s
  return log(s)
end

function logscalar(
  alg::Algorithm,
  tn::AbstractITensorNetwork;
  (cache!)=nothing,
  cache_construction_kwargs=default_cache_construction_kwargs(alg, tn),
  update_cache=isnothing(cache!),
  cache_update_kwargs=default_cache_update_kwargs(alg),
)
  if isnothing(cache!)
    cache! = Ref(cache(alg, tn; cache_construction_kwargs...))
  end

  if update_cache
    cache![] = update(cache![]; cache_update_kwargs...)
  end

  numerator_terms, denominator_terms = scalar_factors_quotient(cache![])
  numerator_terms =
    any(t -> real(t) < 0, numerator_terms) ? complex.(numerator_terms) : numerator_terms
  denominator_terms = if any(t -> real(t) < 0, denominator_terms)
    complex.(denominator_terms)
  else
    denominator_terms
  end

  any(iszero, denominator_terms) && return -Inf
  return sum(log.(numerator_terms)) - sum(log.((denominator_terms)))
end

function ITensors.scalar(alg::Algorithm, tn::AbstractITensorNetwork; kwargs...)
  return exp(logscalar(alg, tn; kwargs...))
end
