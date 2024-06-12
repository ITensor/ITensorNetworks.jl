using ITensorNetworks: update_factors, edge_tag
using ITensors: uniqueinds, factorize_svd, factorize
using LinearAlgebra: norm

function renormalize_update_norm_cache(
  ψ::ITensorNetwork,
  ψOψ_bpcs::Vector{<:BeliefPropagationCache},
  ψIψ_bpc::BeliefPropagationCache;
  cache_update_kwargs,
)
  ψ = copy(ψ)
  ψIψ_bpc = delete_messages(ψIψ_bpc)
  ψOψ_bpcs = delete_messages.(ψOψ_bpcs)
  ψIψ_bpc = update(ψIψ_bpc; cache_update_kwargs...)
  ψIψ_bpc = renormalize_messages(ψIψ_bpc)
  qf = tensornetwork(ψIψ_bpc)

  for v in vertices(ψ)
    v_ket, v_bra = ket_vertex(qf, v), bra_vertex(qf, v)
    pv = only(partitionvertices(ψIψ_bpc, [v_ket]))
    vn = region_scalar(ψIψ_bpc, pv)
    state = (1.0 / sqrt(vn)) * ψ[v]
    state_dag = copy(dag(state))
    state_dag = replaceinds(
      state_dag, inds(state_dag), dual_index_map(qf).(inds(state_dag))
    )
    vertices_states = Dictionary([v_ket, v_bra], [state, state_dag])
    ψOψ_bpcs = update_factors.(ψOψ_bpcs, (vertices_states,))
    ψIψ_bpc = update_factors(ψIψ_bpc, vertices_states)
    ψ[v] = state
  end

  return ψ, ψOψ_bpcs, ψIψ_bpc
end

#TODO: Add support for nsites = 2
function bp_inserter(
  ψ::AbstractITensorNetwork,
  ψOψ_bpcs::Vector{<:BeliefPropagationCache},
  ψIψ_bpc::BeliefPropagationCache,
  state::ITensor,
  region;
  nsites::Int64=1,
  bp_update_kwargs,
  kwargs...,
)
  ψ = copy(ψ)
  form_network = tensornetwork(ψIψ_bpc)
  if length(region) == 1
    states = ITensor[state]
  else
    error("Region lengths of more than 1 not supported for now")
  end

  for (state, v) in zip(states, region)
    ψ[v] = state
    state_dag = copy(ψ[v])
    state_dag = replaceinds(
      dag(state_dag), inds(state_dag), dual_index_map(form_network).(inds(state_dag))
    )
    form_bra_v, form_op_v, form_ket_v = bra_vertex(form_network, v),
    operator_vertex(form_network, v),
    ket_vertex(form_network, v)
    vertices_states = Dictionary([form_ket_v, form_bra_v], [state, state_dag])
    ψOψ_bpcs = update_factors.(ψOψ_bpcs, (vertices_states,))
    ψIψ_bpc = update_factors(ψIψ_bpc, vertices_states)
  end

  ψ, ψOψ_bpcs, ψIψ_bpc = renormalize_update_norm_cache(
    ψ, ψOψ_bpcs, ψIψ_bpc; cache_update_kwargs=bp_update_kwargs
  )

  return ψ, ψOψ_bpcs, ψIψ_bpc
end
