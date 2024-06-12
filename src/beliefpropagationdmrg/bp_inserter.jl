using ITensorNetworks: update_factors

#TODO: Add support for nsites = 2
function bp_inserter(ψ::AbstractITensorNetwork, ψOψ_bpcs::Vector{<:BeliefPropagationCache}, 
    ψIψ_bpc::BeliefPropagationCache, state::ITensor, region; nsites::Int64 = 1, kwargs...)

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

    return ψ, ψOψ_bpcs, ψIψ_bpc
end