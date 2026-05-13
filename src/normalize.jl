using Dictionaries: Dictionary
using Graphs: is_tree
using LinearAlgebra: normalize
using NamedGraphs.PartitionedGraphs: PartitionedGraph, quotient_graph

function rescale(tn::AbstractITensorNetwork; alg = "exact", kwargs...)
    return rescale(Algorithm(alg), tn; kwargs...)
end

function rescale(alg::Algorithm"exact", tn::AbstractITensorNetwork; kwargs...)
    logn = logscalar(alg, tn; kwargs...)
    c = inv(exp(logn / length(vertices(tn))))
    return map(t -> c * t, tn)
end

function rescale(
        alg::Algorithm"bp",
        tn::AbstractFormNetwork,
        args...;
        (cache!) = nothing,
        cache_construction_kwargs = (;),
        update_cache = isnothing(cache!),
        cache_update_kwargs = (;),
        kwargs...
    )
    if isnothing(cache!)
        pv = get(
            cache_construction_kwargs, :partitioned_vertices,
            default_partitioned_vertices(tn)
        )
        ptn = PartitionedGraph(tn, pv)
        messages = get(cache_construction_kwargs, :messages, nothing)
        if isnothing(messages)
            messages =
                is_tree(quotient_graph(ptn)) ? Dictionary() : identity_messages(tn, ptn)
        end
        cache! = Ref(BeliefPropagationCache(ptn; messages))
    end

    if update_cache
        cache![] = update(cache![]; cache_update_kwargs...)
    end

    cache![] = rescale(cache![], args...; kwargs...)

    return tensornetwork(cache![])
end

function rescale(
        alg::Algorithm,
        tn::AbstractITensorNetwork,
        args...;
        (cache!) = nothing,
        cache_construction_kwargs = default_cache_construction_kwargs(alg, tn),
        update_cache = isnothing(cache!),
        cache_update_kwargs = (;),
        kwargs...
    )
    if isnothing(cache!)
        cache! = Ref(cache(alg, tn; cache_construction_kwargs...))
    end

    if update_cache
        cache![] = update(cache![]; cache_update_kwargs...)
    end

    cache![] = rescale(cache![], args...; kwargs...)

    return tensornetwork(cache![])
end

"""
    normalize(tn::AbstractITensorNetwork; alg="exact", kwargs...) -> AbstractITensorNetwork

Return a copy of `tn` rescaled so that `norm(tn) ≈ 1`.

The rescaling is distributed evenly across all tensors in the network (each tensor is
multiplied by the same scalar factor).

# Keyword Arguments

  - `alg="exact"`: Normalization algorithm. `"exact"` contracts ⟨ψ|ψ⟩ exactly;
    `"bp"` uses belief propagation for large networks.

See also: `norm`, [`inner`](@ref ITensorNetworks.inner).
"""
function LinearAlgebra.normalize(tn::AbstractITensorNetwork; alg = "exact", kwargs...)
    return normalize(Algorithm(alg), tn; kwargs...)
end

function LinearAlgebra.normalize(
        alg::Algorithm"exact", tn::AbstractITensorNetwork; kwargs...
    )
    logn = logscalar(alg, inner_network(tn, tn); kwargs...)
    c = inv(exp(logn / (2 * length(vertices(tn)))))
    return map(t -> c * t, tn)
end

function LinearAlgebra.normalize(
        alg::Algorithm"bp",
        tn::AbstractITensorNetwork;
        (cache!) = nothing,
        update_cache = isnothing(cache!),
        cache_update_kwargs = (;),
        cache_construction_kwargs = (;)
    )
    norm_tn = inner_network(tn, tn)
    if isnothing(cache!)
        pv = get(
            cache_construction_kwargs, :partitioned_vertices,
            default_partitioned_vertices(norm_tn)
        )
        ptn = PartitionedGraph(norm_tn, pv)
        messages = get(cache_construction_kwargs, :messages, nothing)
        if isnothing(messages)
            messages = if is_tree(quotient_graph(ptn))
                Dictionary()
            else
                identity_messages(norm_tn, ptn)
            end
        end
        cache! = Ref(BeliefPropagationCache(ptn; messages))
    end
    vs = collect(vertices(tn))
    verts = vcat([ket_vertex(norm_tn, v) for v in vs], [bra_vertex(norm_tn, v) for v in vs])
    norm_tn = rescale(alg, norm_tn; verts, cache!, update_cache, cache_update_kwargs)

    return ket_network(norm_tn)
end
