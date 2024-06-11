using ITensorNetworks: update_factors

function normalize_state_update_caches(ψ::ITensorNetwork, ψIψ::BeliefPropagationCache, ψOψs::Vector{<:BeliefPropagationCache})
    qf = tensornetwork(ψIψ)
    L = length(vertices(ψ))
    cur_norm = scalar(ψIψ)
    rescale_coeff = cur_norm^(-1/(2*L))
    ψ = copy(ψ)
    for v in vertices(ψ)
        form_bra_v, form_ket_v = bra_vertex(qf, v), ket_vertex(qf, v)
        state = ψ[v] * rescale_coeff
        ψ[v] = state
        state_dag = copy(ψ[v])
        state_dag = replaceinds(dag(state_dag), inds(state_dag), dual_index_map(qf).(inds(state_dag)))
        vertices_states = Dictionary([form_ket_v, form_bra_v], [state, state_dag])
        ψIψ = update_factors(ψIψ, vertices_states)
        ψOψs = update_factors.(ψOψs, (vertices_states, ))
    end

    return ψ, ψIψ, ψOψs
end

#TODO: Add support for nsites = 2
function bp_inserter(ψ::AbstractITensorNetwork, ψOψ_bpcs::Vector{<:BeliefPropagationCache}, 
    ψIψ_bpc::BeliefPropagationCache, state::ITensor, region; normalize_state = true, cache_update_kwargs, nsites::Int64 = 1, kwargs...)

    @assert nsites == 1
    ψ = copy(ψ)
    form_network = tensornetwork(ψIψ_bpc)
    states = ITensor[state]

    for (state, v) in zip(states, region)
        ψ[v] = state
        state_dag = copy(ψ[v])
        state_dag = replaceinds(dag(state_dag), inds(state_dag), dual_index_map(form_network).(inds(state_dag)))
        form_bra_v, form_op_v, form_ket_v = bra_vertex(form_network, v), operator_vertex(form_network, v), ket_vertex(form_network, v)
        vertices_states = Dictionary([form_ket_v, form_bra_v], [state, state_dag])
        ψOψ_bpcs = update_factors.(ψOψ_bpcs, (vertices_states,))
        ψIψ_bpc = update_factors(ψIψ_bpc, vertices_states)
    end

    ψIψ_bpc = update(ψIψ_bpc; cache_update_kwargs...)
    if normalize_state
        ψ, ψIψ_bpc, ψOψ_bpcs = normalize_state_update_caches(ψ, ψIψ_bpc, ψOψ_bpcs)
    end

    return ψ, ψOψ_bpcs, ψIψ_bpc
end