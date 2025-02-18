using ITensors: inner, scalar
using ITensorMPS: ITensorMPS, loginner
using LinearAlgebra: norm, norm_sqr

default_contract_alg(tns::Tuple) = "bp"

function ITensors.inner(
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  alg=default_contract_alg((ϕ, ψ)),
  kwargs...,
)
  return inner(Algorithm(alg), ϕ, ψ; kwargs...)
end

function ITensors.inner(
  ϕ::AbstractITensorNetwork,
  A::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  alg=default_contract_alg((ϕ, A, ψ)),
  kwargs...,
)
  return inner(Algorithm(alg), ϕ, A, ψ; kwargs...)
end

function ITensors.inner(
  alg::Algorithm"exact",
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  sequence=nothing,
  contraction_sequence_kwargs=(;),
  kwargs...,
)
  tn = inner_network(ϕ, ψ; kwargs...)
  if isnothing(sequence)
    sequence = contraction_sequence(tn; contraction_sequence_kwargs...)
  end
  return scalar(tn; sequence)
end

function ITensors.inner(
  alg::Algorithm"exact",
  ϕ::AbstractITensorNetwork,
  A::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  sequence=nothing,
  contraction_sequence_kwargs=(;),
  kwargs...,
)
  tn = inner_network(ϕ, A, ψ; kwargs...)
  if isnothing(sequence)
    sequence = contraction_sequence(tn; contraction_sequence_kwargs...)
  end
  return scalar(tn; sequence)
end

function ITensorMPS.loginner(
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  alg=default_contract_alg((ϕ, ψ)),
  kwargs...,
)
  return loginner(Algorithm(alg), ϕ, ψ; kwargs...)
end

function ITensorMPS.loginner(
  ϕ::AbstractITensorNetwork,
  A::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  alg=default_contract_alg((ϕ, A, ψ)),
  kwargs...,
)
  return loginner(Algorithm(alg), ϕ, A, ψ; kwargs...)
end

function ITensorMPS.loginner(
  alg::Algorithm"exact", ϕ::AbstractITensorNetwork, ψ::AbstractITensorNetwork; kwargs...
)
  return log(inner(alg, ϕ, ψ); kwargs...)
end

function ITensorMPS.loginner(
  alg::Algorithm"exact",
  ϕ::AbstractITensorNetwork,
  A::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  kwargs...,
)
  return log(inner(alg, ϕ, A, ψ); kwargs...)
end

function ITensorMPS.loginner(
  alg::Algorithm,
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  dual_link_index_map=sim,
  kwargs...,
)
  tn = inner_network(ϕ, ψ; dual_link_index_map)
  return logscalar(alg, tn; kwargs...)
end

function ITensorMPS.loginner(
  alg::Algorithm,
  ϕ::AbstractITensorNetwork,
  A::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  dual_link_index_map=sim,
  kwargs...,
)
  tn = inner_network(ϕ, A, ψ; dual_link_index_map)
  return logscalar(alg, tn; kwargs...)
end

function ITensors.inner(
  alg::Algorithm,
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  dual_link_index_map=sim,
  kwargs...,
)
  tn = inner_network(ϕ, ψ; dual_link_index_map)
  return scalar(alg, tn; kwargs...)
end

function ITensors.inner(
  alg::Algorithm,
  ϕ::AbstractITensorNetwork,
  A::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  dual_link_index_map=sim,
  kwargs...,
)
  tn = inner_network(ϕ, A, ψ; dual_link_index_map)
  return scalar(alg, tn; kwargs...)
end

# TODO: rename `sqnorm` to match https://github.com/JuliaStats/Distances.jl,
# or `norm_sqr` to match `LinearAlgebra.norm_sqr`
LinearAlgebra.norm_sqr(ψ::AbstractITensorNetwork; kwargs...) = inner(ψ, ψ; kwargs...)

function LinearAlgebra.norm(ψ::AbstractITensorNetwork; kwargs...)
  return sqrt(abs(real(norm_sqr(ψ; kwargs...))))
end
