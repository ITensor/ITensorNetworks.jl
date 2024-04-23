using Dictionaries: Dictionary, set!
using ITensors: Op, op, contract, siteinds, which_op
using ITensors.ITensorMPS: ITensorMPS, expect

default_expect_alg() = "bp"

expect_network(ψ::AbstractITensorNetwork; kwargs...) = inner_network(ψ, ψ; kwargs...)

function ITensorMPS.expect(
  ψ::AbstractITensorNetwork, args...; alg=default_expect_alg(), kwargs...
)
  return expect(Algorithm(alg), ψ, args...; kwargs...)
end

function ITensorMPS.expect(alg::Algorithm, ψ::AbstractITensorNetwork, op::String; kwargs...)
  return expect(alg, ψ, op, vertices(ψ); kwargs...)
end

function ITensorMPS.expect(
  alg::Algorithm, ψ::AbstractITensorNetwork, op::String, vertex; kwargs...
)
  return expect(alg, ψ, op, [vertex]; kwargs...)
end

#TODO: What to call this function?!
function expect_internal(
  ψ::AbstractITensorNetwork,
  ψIψ::AbstractFormNetwork,
  ops::Vector{String},
  vertices::Vector;
  contract_kwargs=(;),
  kwargs...,
)
  @assert length(vertices) == length(ops)
  op_vertices = [operator_vertex(ψIψ, v) for v in vertices]
  s = siteinds(ψ)
  operators = ITensor[ITensors.op(op, s[vertices[i]]) for (i, op) in enumerate(ops)]

  ψIψ_vs = ITensor[ψIψ[op_vertex] for op_vertex in op_vertices]
  ∂ψIψ_∂v = environment(ψIψ, vertices; vertex_mapping_function=operator_vertices, kwargs...)
  numerator = contract(vcat(∂ψIψ_∂v, operators); contract_kwargs...)[]
  denominator = contract(vcat(∂ψIψ_∂v, ψIψ_vs); contract_kwargs...)[]

  return numerator / denominator
end

function ITensorMPS.expect(
  alg::Algorithm,
  ψ::AbstractITensorNetwork,
  op::String,
  vertices::Vector;
  (cache!)=nothing,
  update_cache=isnothing(cache!),
  cache_update_kwargs=default_cache_update_kwargs(cache!),
  cache_construction_function=tn ->
    cache(alg, tn; default_cache_construction_kwargs(alg, tn)...),
  kwargs...,
)
  ψIψ = expect_network(ψ)
  if isnothing(cache!)
    cache! = Ref(cache_construction_function(ψIψ))
  end

  if update_cache
    cache![] = update(cache![]; cache_update_kwargs...)
  end

  return map(
    vertex ->
      expect_internal(ψ, ψIψ, [op], [vertex]; alg, cache!, update_cache=false, kwargs...),
    vertices,
  )
end

function ITensorMPS.expect(
  alg::Algorithm"exact", ψ::AbstractITensorNetwork, op::String, vertices::Vector; kwargs...
)
  ψIψ = expect_network(ψ)
  return map(vertex -> expect_internal(ψ, ψIψ, [op], [vertex]; alg, kwargs...), vertices)
end
