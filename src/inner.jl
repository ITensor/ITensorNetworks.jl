default_inner_partitioned_vertices(tn) = group(v -> first(v), vertices(tn))
default_algorithm(tns::Vector) = all(is_tree.(tns)) ? "bp" : "exact"

function inner(
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  alg=default_algorithm([ϕ, ψ]),
  kwargs...,
)
  return inner(Algorithm(alg), ϕ, ψ; kwargs...)
end

function inner(
  A::AbstractITensorNetwork,
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  alg=default_algorithm([A, ϕ, ψ]),
  kwargs...,
)
  return inner(Algorithm(alg), A, ϕ, ψ; kwargs...)
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
  A::AbstractITensorNetwork,
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  sequence=nothing,
  contraction_sequence_kwargs=(;),
  site_index_map=prime,
  kwargs...,
)
  tn = flatten_networks(site_index_map(dag(ϕ); links=[]), A, ψ; kwargs...)
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
  alg::Algorithm"exact", ϕ::AbstractITensorNetwork, ψ::AbstractITensorNetwork; kwargs...
)
  return log(inner(alg, ϕ, ψ); kwargs...)
end

function loginner(
  alg::Algorithm"bp",
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  partitioned_verts=default_inner_partitioned_vertices,
  link_index_map=sim,
  kwargs...,
)
  tn = disjoint_union("bra" => link_index_map(dag(ϕ); sites=[]), "ket" => ψ)
  return logscalar(alg, tn; partitioned_vertices=partitioned_verts(tn), kwargs...)
end

function loginner(
  alg::Algorithm"bp",
  A::AbstractITensorNetwork,
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  partitioned_verts=default_inner_partitioned_vertices,
  site_index_map=prime,
  link_index_map=sim,
  kwargs...,
)
  ϕ_dag = link_index_map(site_index_map(dag(ϕ); links=[]); sites=[])
  tn = disjoint_union("operator" => A, "bra" => ϕ_dag, "ket" => ψ)
  return logscalar(alg, tn; partitioned_vertices=partitioned_verts(tn), kwargs...)
end

function inner(
  alg::Algorithm"bp", ϕ::AbstractITensorNetwork, ψ::AbstractITensorNetwork; kwargs...
)
  return exp(loginner(alg, ϕ, ψ; kwargs...))
end

function inner(
  alg::Algorithm"bp",
  A::AbstractITensorNetwork,
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  kwargs...,
)
  return exp(loginner(alg, A, ϕ, ψ; kwargs...))
end

# TODO: rename `sqnorm` to match https://github.com/JuliaStats/Distances.jl,
# or `norm_sqr` to match `LinearAlgebra.norm_sqr`
function norm_sqr(ψ::AbstractITensorNetwork; kwargs...)
  ψ_mapped = sim(ψ; sites=[])
  return inner(ψ, ψ_mapped; kwargs...)
end
