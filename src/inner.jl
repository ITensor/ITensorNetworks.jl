default_inner_partitioned_vertices(tn) = group(v -> first(v), vertices(tn))

function inner(ϕ::AbstractITensorNetwork,
    ψ::AbstractITensorNetwork; alg = "exact", kwargs...)
    return inner(Algorithm(alg), ϕ, ψ; kwargs...)
end

function inner(alg::Algorithm"exact",
    ϕ::AbstractITensorNetwork,
    ψ::AbstractITensorNetwork;
    sequence=nothing,
    combine_linkinds = true,
    flatten = true,
    contraction_sequence_kwargs=(;),
    )
    tn = flatten_networks(dag(ϕ), ψ; combine_linkinds, flatten)
    if isnothing(sequence)
        sequence = contraction_sequence(tn; contraction_sequence_kwargs...)
    end
    return scalar(tn; sequence)
end

function inner(alg::Algorithm"exact",
    A::AbstractITensorNetwork,
    ϕ::AbstractITensorNetwork,
    ψ::AbstractITensorNetwork;
    site_index_map = default_index_map,
    combine_linkinds,
    flatten,
    kwargs...,
    )
    tn = flatten_networks(site_index_map(dag(ϕ); links = []), A, ψ; combine_linkinds, flatten)
    if isnothing(sequence)
        sequence = contraction_sequence(tn; contraction_sequence_kwargs...)
    end
    return scalar(tn; sequence)
end

loginner(ϕ::AbstractITensorNetwork, ψ::AbstractITensorNetwork; alg = "exact", kwargs...) = loginner(Algorithm(alg), ϕ, ψ; kwargs...)
loginner(alg::Algorithm"exact", ϕ::AbstractITensorNetwork, ψ::AbstractITensorNetwork; kwargs...) = log(inner(alg, ϕ, ψ); kwargs...)

function loginner(alg::Algorithm"bp", ϕ::AbstractITensorNetwork, ψ::AbstractITensorNetwork; 
    partitioned_vertices = default_inner_partitioned_vertices,
    kwargs...)
    ϕ_dag = sim(dag(ϕ); sites = [])
    tn = disjoint_union("bra" => ϕ_dag, "ket" => ψ)
    return logscalar(alg, tn; partitioned_vertices, kwargs...)
end

function loginner(alg::Algorithm"bp", A::AbstractITensorNetwork, ϕ::AbstractITensorNetwork, ψ::AbstractITensorNetwork; 
    partitioned_vertices = default_inner_partitioned_vertices, site_index_map = default_index_map,
    kwargs...)
    ϕ_dag = site_index_map(sim(dag(ϕ); sites = []); links = [])
    tn = disjoint_union("operator" => A, "bra" => ϕ_dag, "ket" => ψ)
    return logscalar(alg, tn; partitioned_vertices, kwargs...)
end

inner(alg::Algorithm"bp", ϕ::AbstractITensorNetwork,
    ψ::AbstractITensorNetwork; kwargs...) = exp(loginner(alg, ϕ, ψ; kwargs...))



