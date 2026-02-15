using ITensors.NDTensors: @Algorithm_str, Algorithm, NDTensors, contract
using ITensors: ITensor, scalar
using LinearAlgebra: normalize!
using NamedGraphs.OrdinalIndexing: th
using NamedGraphs: NamedGraphs

function NDTensors.contract(tn::AbstractITensorNetwork; alg = "exact", kwargs...)
    return contract(Algorithm(alg), tn; kwargs...)
end

function NDTensors.contract(
        alg::Algorithm"exact",
        tn::AbstractITensorNetwork;
        contraction_sequence_kwargs = (;),
        sequence = contraction_sequence(tn; contraction_sequence_kwargs...),
        kwargs...
    )
    sequence_linear_index = deepmap(v -> NamedGraphs.vertex_positions(tn)[v], sequence)
    ts = map(v -> tn[v], (1:nv(tn))th)
    return contract(ts; sequence = sequence_linear_index, kwargs...)
end

function NDTensors.contract(
        alg::Union{Algorithm"density_matrix", Algorithm"ttn_svd"},
        tn::AbstractITensorNetwork;
        output_structure::Function = path_graph_structure,
        kwargs...
    )
    return contract_approx(alg, tn, output_structure; kwargs...)
end

function ITensors.scalar(alg::Algorithm"exact", tn::AbstractITensorNetwork; kwargs...)
    return contract(alg, tn; kwargs...)[]
end

function ITensors.scalar(tn::AbstractITensorNetwork; alg = "exact", kwargs...)
    return scalar(Algorithm(alg), tn; kwargs...)
end

function logscalar(tn::AbstractITensorNetwork; alg = "exact", kwargs...)
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
        (cache!) = nothing,
        cache_construction_kwargs = default_cache_construction_kwargs(alg, tn),
        update_cache = isnothing(cache!),
        cache_update_kwargs = (;)
    )
    if isnothing(cache!)
        cache! = Ref(cache(alg, tn; cache_construction_kwargs...))
    end

    if update_cache
        cache![] = update(cache![]; cache_update_kwargs...)
    end

    return logscalar(cache![])
end

function ITensors.scalar(alg::Algorithm, tn::AbstractITensorNetwork; kwargs...)
    return exp(logscalar(alg, tn; kwargs...))
end
