using Dictionaries: Dictionary, set!
using ITensors: Op, op, contract, siteinds, which_op
using ITensors.ITensorMPS: ITensorMPS, expect

default_expect_alg() = "bp"

#Don't need this...
expect_network(ψ::AbstractITensorNetwork; kwargs...) = inner_network(ψ, ψ; kwargs...)

function ITensorMPS.expect(
  ψ::AbstractITensorNetwork, args...; alg=default_expect_alg(), kwargs...
)
  return expect(Algorithm(alg), ψ, args...; kwargs...)
end

#TODO: What to name this? It calculates the expectation value <Prod{op in ops}> over ψ 
# where you pass the norm network which is constructed elsewhere. All expect() calls should pass
# down to this internal function. The alg is passed along to `environment`, which is where the work is 
# actually done.
# Get siteinds out of psi I psi instead?
function expect_internal(ψIψ::AbstractFormNetwork, op::Op; contract_kwargs=(;), kwargs...)
  v = only(op.sites)

  ψIψ_v = ψIψ[operator_vertex(ψIψ, v)]
  s = commonind(ψIψ[ket_vertex(ψIψ, v)], ψIψ_v)
  operator = ITensors.op(op.which_op, s)
  ∂ψIψ_∂v = environment(ψIψ, [v]; vertex_mapping_function=operator_vertices, kwargs...)
  numerator = contract(vcat(∂ψIψ_∂v, operator); contract_kwargs...)[]
  denominator = contract(vcat(∂ψIψ_∂v, ψIψ_v); contract_kwargs...)[]

  return numerator / denominator
end

#Remove type constraint on ops_collection
function ITensorMPS.expect(
  alg::Algorithm,
  ψ::AbstractITensorNetwork,
  ops::Vector{Op};
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
    op -> expect_internal(ψIψ, op; alg, cache!, update_cache=false, kwargs...), ops
  )
end

function ITensorMPS.expect(
  alg::Algorithm"exact", ψ::AbstractITensorNetwork, ops::Vector{Op}; kwargs...
)
  ψIψ = expect_network(ψ)
  return map(op -> expect_internal(ψIψ, op; alg, kwargs...), ops)
end

function ITensorMPS.expect(alg::Algorithm, ψ::AbstractITensorNetwork, op::Op; kwargs...)
  return expect(alg, ψ, [op]; kwargs...)
end

function ITensorMPS.expect(
  alg::Algorithm, ψ::AbstractITensorNetwork, op::String, vertex; kwargs...
)
  return expect(alg, ψ, Op(op, vertex); kwargs...)
end

function ITensorMPS.expect(
  alg::Algorithm, ψ::AbstractITensorNetwork, op::String, vertices::Vector; kwargs...
)
  return expect(alg, ψ, [Op(op, vertex) for vertex in vertices]; kwargs...)
end

function ITensorMPS.expect(alg::Algorithm, ψ::AbstractITensorNetwork, op::String; kwargs...)
  return expect(alg, ψ, op, vertices(ψ); kwargs...)
end
