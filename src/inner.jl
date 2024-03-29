default_algorithm(tns::Tuple) = "bp"

#Default for map_linkinds should be sim.

function inner(
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  alg=default_algorithm((ϕ, ψ)),
  kwargs...,
)
  return inner(Algorithm(alg), ϕ, ψ; kwargs...)
end

function inner(
  ϕ::AbstractITensorNetwork,
  A::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  alg=default_algorithm((ϕ, A, ψ)),
  kwargs...,
)
  return inner(Algorithm(alg), ϕ, A, ψ; kwargs...)
end

function inner(
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

function inner(
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

function loginner(
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  alg=default_algorithm(ϕ, ψ),
  kwargs...,
)
  return loginner(Algorithm(alg), ϕ, ψ; kwargs...)
end

function loginner(
  ϕ::AbstractITensorNetwork,
  A::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  alg=default_algorithm(ϕ, ψ),
  kwargs...,
)
  return loginner(Algorithm(alg), ϕ, A, ψ; kwargs...)
end

function loginner(
  alg::Algorithm"exact", ϕ::AbstractITensorNetwork, ψ::AbstractITensorNetwork; kwargs...
)
  return log(inner(alg, ϕ, ψ); kwargs...)
end

function loginner(
  alg::Algorithm"exact",
  ϕ::AbstractITensorNetwork,
  A::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  kwargs...,
)
  return log(inner(alg, ϕ, A, ψ); kwargs...)
end

function loginner(
  alg::Algorithm"bp",
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  dual_link_index_map=sim,
  kwargs...,
)
  tn = inner_network(ϕ, ψ; dual_link_index_map)
  return logscalar(alg, tn; kwargs...)
end

function loginner(
  alg::Algorithm"bp",
  ϕ::AbstractITensorNetwork,
  A::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  dual_link_index_map=sim,
  kwargs...,
)
  tn = inner_network(ϕ, A, ψ; dual_link_index_map)
  return logscalar(alg, tn; kwargs...)
end

function inner(
  alg::Algorithm"bp",
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  dual_link_index_map=sim,
  kwargs...,
)
  tn = inner_network(ϕ, ψ; dual_link_index_map)
  return scalar(alg, tn; kwargs...)
end

function inner(
  alg::Algorithm"bp",
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
norm_sqr(ψ::AbstractITensorNetwork; kwargs...) = inner(ψ, ψ; kwargs...)

function LinearAlgebra.norm(ψ::AbstractITensorNetwork; kwargs...)
  return sqrt(abs(real(norm_sqr(ψ; kwargs...))))
end
