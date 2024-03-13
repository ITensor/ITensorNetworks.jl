default_derivative_algorithm() = "exact"

function derivative(ψ::AbstractITensorNetwork, vertices::Vector; alg = default_derivative_algorithm(), kwargs...)
    return derivative(Algorithm(alg), ψ, vertices; kwargs...)
end

function derivative(::Algorithm"exact", ψ::AbstractITensorNetwork, vertices::Vector; contraction_sequence_alg = "optimal", kwargs...)
    ψ_reduced = Vector{ITensor}(subgraph(ψ, vertices))
    return contract_exact(ψ_reduced; normalize = false, contraction_sequence_alg, kwargs...)
end

function derivative(::Algorithm"bp", ψ::AbstractITensorNetwork, vertices::Vector; (bp_cache!) = nothing, bp_cache_update_kwargs = default_cache_update_kwargs(bp_cache))

    if isnothing(bp_cache!)
        bp_cache! = Ref(default_cache(ψ))
    end
    bp_cache![] = update(bp_cache![]; bp_cache_update_kwargs...)
    return incoming_messages(bp_cache![], vertices)
end
    