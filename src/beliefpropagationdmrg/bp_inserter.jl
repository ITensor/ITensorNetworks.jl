# function bp_inserter(ψ::AbstractITensorNetwork, ψAψ_bpcs::Vector{<:BeliefPropagationCache}, 
#     ψIψ_bpc::BeliefPropagationCache, state::ITensor, region; cache_update_kwargs = (;), kwargs...)
  
#     spec = nothing
  
#     form_network = unpartitioned_graph(partitioned_tensornetwork(ψIψ_bpc))
#     if length(region) == 1
#       states = [state]
#     elseif length(region) == 2
#       v1, v2 = region[1], region[2]
#       e = edgetype(ψ)(v1, v2)
#       pe = partitionedge(ψIψ_bpc, ket_vertex(form_network, v1) => bra_vertex(form_network, v2))
#       stateᵥ₁, stateᵥ₂, spec = factorize_svd(state,uniqueinds(ψ[v1], ψ[v2]); ortho="none", tags=edge_tag(e),kwargs...)
#       states = noprime.([stateᵥ₁, stateᵥ₂])
#       ψIψ_bpc = reset_messages(ψIψ_bpc, [pe, reverse(pe)])
#       ψAψ_bpcs = BeliefPropagationCache[reset_messages(ψAψ_bpc, [pe, reverse(pe)]) for ψAψ_bpc in ψAψ_bpcs]
#       #TODO: Insert spec into the message tensor guess here?!
#     end
  
#     for (i, v) in enumerate(region)
#       state = states[i]
#       state_dag = copy(state)
#       form_bra_v, form_ket_v = bra_vertex(form_network, v), ket_vertex(form_network, v)
#       ψ[v] =state
#       state_dag = replaceinds(dag(state_dag), inds(state_dag), dual_index_map(form_network).(inds(state_dag)))
#       ψAψ_bpcs = BeliefPropagationCache[update_factor(ψAψ_bpc, form_ket_v, state) for ψAψ_bpc in ψAψ_bpcs]
#       ψAψ_bpcs = BeliefPropagationCache[update_factor(ψAψ_bpc, form_bra_v, state_dag) for ψAψ_bpc in ψAψ_bpcs]
#       ψIψ_bpc = update_factor(ψIψ_bpc, form_ket_v, state)
#       ψIψ_bpc = update_factor(ψIψ_bpc, form_bra_v, state_dag)
#     end
  
  
#     ψAψ_bpcs = BeliefPropagationCache[update(ψAψ_bpc; cache_update_kwargs...) for ψAψ_bpc in ψAψ_bpcs]
  
#     ψIψ_bpc = update(ψIψ_bpc; cache_update_kwargs...)
  
#     return ψ, ψAψ_bpcs, ψIψ_bpc, spec, (; eigvals=[0.0])
#   end

function bp_inserter_one_site(ψ::AbstractITensorNetwork, ψAψ_bpcs::Vector{<:BeliefPropagationCache}, 
    ψIψ_bpc::BeliefPropagationCache, state::ITensor, region; cache_update_kwargs = (;), kwargs...)

    spec = nothing

    ψ = copy(ψ)
    form_network = unpartitioned_graph(partitioned_tensornetwork(ψIψ_bpc))
    @assert length(region) == 1
    states = [state]

    for (i, v) in enumerate(region)
        state = states[i]
        state_dag = copy(state)
        form_bra_v, form_op_v, form_ket_v = bra_vertex(form_network, v), operator_vertex(form_network, v), ket_vertex(form_network, v)
        ψ[v] =state
        state_dag = replaceinds(dag(state_dag), inds(state_dag), dual_index_map(form_network).(inds(state_dag)))
        ψAψ_bpcs = BeliefPropagationCache[update_factor(ψAψ_bpc, form_ket_v, state) for ψAψ_bpc in ψAψ_bpcs]
        ψAψ_bpcs = BeliefPropagationCache[update_factor(ψAψ_bpc, form_bra_v, state_dag) for ψAψ_bpc in ψAψ_bpcs]
        ψIψ_bpc = update_factor(ψIψ_bpc, form_ket_v, state)
        ψIψ_bpc = update_factor(ψIψ_bpc, form_bra_v, state_dag)
    end

    return ψ, ψAψ_bpcs, ψIψ_bpc, spec, (; eigvals=[0.0])
end