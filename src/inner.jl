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
  ϕ::AbstractITensorNetwork,
  A::AbstractITensorNetwork,
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
  ϕ::AbstractITensorNetwork,
  A::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  sequence=nothing,
  contraction_sequence_kwargs=(;),
  site_index_map=prime,
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
  return loginner(Algorithm(alg), A, ϕ, ψ; kwargs...)
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
  partitioned_verts=default_inner_partitioned_vertices,
  link_index_map=sim,
  kwargs...,
)
  ϕ_dag = map_inds(link_index_map, dag(ϕ); sites=[])
  tn = disjoint_union("bra" => ϕ_dag, "ket" => ψ)
  return logscalar(alg, tn; partitioned_vertices=partitioned_verts(tn), kwargs...)
end

function loginner(
  alg::Algorithm"bp",
  ϕ::AbstractITensorNetwork,
  A::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  partitioned_verts=default_inner_partitioned_vertices,
  link_index_map=sim,
  kwargs...,
)
  ϕ_dag = map_inds(link_index_map, dag(ϕ); sites=[])
  tn = disjoint_union("operator" => A, "bra" => ϕ_dag, "ket" => ψ)
  return logscalar(alg, tn; partitioned_vertices=partitioned_verts(tn), kwargs...)
end

function inner(
  alg::Algorithm"bp",
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  partitioned_verts=default_inner_partitioned_vertices,
  link_index_map=prime,
  kwargs...,
)
  ϕ_dag = map_inds(link_index_map, dag(ϕ); sites=[])
  tn = disjoint_union("bra" => ϕ_dag, "ket" => ψ)
  return scalar(alg, tn; partitioned_vertices=partitioned_verts(tn), kwargs...)
end

function inner(
  alg::Algorithm"bp",
  A::AbstractITensorNetwork,
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  partitioned_verts=default_inner_partitioned_vertices,
  link_index_map=prime,
  kwargs...,
)
  ϕ_dag = map_inds(link_index_map, dag(ϕ); sites=[])
  tn = disjoint_union("operator" => A, "bra" => ϕ_dag, "ket" => ψ)
  return scalar(alg, tn; partitioned_vertices=partitioned_verts(tn), kwargs...)
end

# TODO: rename `sqnorm` to match https://github.com/JuliaStats/Distances.jl,
# or `norm_sqr` to match `LinearAlgebra.norm_sqr`
norm_sqr(ψ::AbstractITensorNetwork; kwargs...) = inner(ψ, ψ; kwargs...)

function norm(ψ::AbstractITensorNetwork; kwargs...)
  return sqrt(abs(real(norm_sqr(ψ; kwargs...))))
end
