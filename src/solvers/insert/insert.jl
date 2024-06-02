# Here extract_local_tensor and insert_local_tensor
# are essentially inverse operations, adapted for different kinds of 
# algorithms and networks.

# TODO: use dense TTN constructor to make this more general.
function default_inserter(
  state::AbstractTTN,
  phi::ITensor,
  region,
  ortho_vert;
  normalize=false,
  maxdim=nothing,
  mindim=nothing,
  cutoff=nothing,
  internal_kwargs,
)
  spec = nothing
  other_vertex = setdiff(support(region), [ortho_vert])
  if !isempty(other_vertex)
    v = only(other_vertex)
    e = edgetype(state)(ortho_vert, v)
    indsTe = inds(state[ortho_vert])
    L, phi, spec = factorize(phi, indsTe; tags=tags(state, e), maxdim, mindim, cutoff)
    state[ortho_vert] = L
  else
    v = ortho_vert
  end
  state[v] = phi
  state = set_ortho_region(state, [v])
  normalize && (state[v] /= norm(state[v]))
  return state, spec
end

function default_inserter(
  state::AbstractTTN,
  phi::ITensor,
  region::NamedEdge,
  ortho;
  normalize=false,
  maxdim=nothing,
  mindim=nothing,
  cutoff=nothing,
  internal_kwargs,
)
  v = only(setdiff(support(region), [ortho]))
  state[v] *= phi
  state = set_ortho_region(state, [v])
  return state, nothing
end

function bp_inserter(ψ::AbstractITensorNetwork, ψAψ_bpcs::Vector{<:BeliefPropagationCache}, 
  ψIψ_bpc::BeliefPropagationCache, state::ITensor, region; cache_update_kwargs = (;), kwargs...)

  spec = nothing

  form_network = unpartitioned_graph(partitioned_tensornetwork(ψIψ_bpc))
  ψAψ_bpcs = BeliefPropagationCache[reset_messages(ψAψ_bpc) for ψAψ_bpc in ψAψ_bpcs]
  ψIψ_bpc = reset_messages(ψIψ_bpc)
  @show messages(ψIψ_bpc)
  if length(region) == 1
    states = [state]
  elseif length(region) == 2
    v1, v2 = region[1], region[2]
    e = edgetype(ψ)(v1, v2)
    pe = partitionedge(ψIψ_bpc, ket_vertex(form_network, v1) => bra_vertex(form_network, v2))
    stateᵥ₁, stateᵥ₂, spec = factorize_svd(state,uniqueinds(ψ[v1], ψ[v2]); ortho="none", tags=edge_tag(e),kwargs...)
    states = noprime.([stateᵥ₁, stateᵥ₂])
    #TODO: Insert spec into the message tensor guess here?!
  end

  for (i, v) in enumerate(region)
    state = states[i]
    state_dag = copy(state)
    form_bra_v, form_ket_v = bra_vertex(form_network, v), ket_vertex(form_network, v)
    ψ[v] =state
    state_dag = replaceinds(dag(state_dag), inds(state_dag), dual_index_map(form_network).(inds(state_dag)))
    ψAψ_bpcs = BeliefPropagationCache[update_factor(ψAψ_bpc, form_ket_v, state) for ψAψ_bpc in ψAψ_bpcs]
    ψAψ_bpcs = BeliefPropagationCache[update_factor(ψAψ_bpc, form_bra_v, state_dag) for ψAψ_bpc in ψAψ_bpcs]
    ψIψ_bpc = update_factor(ψIψ_bpc, form_ket_v, state)
    ψIψ_bpc = update_factor(ψIψ_bpc, form_bra_v, state_dag)
  end


  ψAψ_bpcs = BeliefPropagationCache[update(ψAψ_bpc; cache_update_kwargs...) for ψAψ_bpc in ψAψ_bpcs]

  ψIψ_bpc = update(ψIψ_bpc; cache_update_kwargs...)

  updated_ψIψ = unpartitioned_graph(partitioned_tensornetwork(ψIψ_bpc))
  numerator_terms = [scalar(unpartitioned_graph(partitioned_tensornetwork(ψAψ_bpc)); cache! = Ref(ψAψ_bpc), alg = "bp") for ψAψ_bpc in ψAψ_bpcs]
  eigval = sum(numerator_terms) / scalar(updated_ψIψ; cache! = Ref(ψIψ_bpc), alg = "bp")
  @show eigval
  @show scalar(only(ψAψ_bpcs); alg = "exact") / scalar(ψIψ_bpc; alg = "exact")
  return ψ, ψAψ_bpcs, ψIψ_bpc, spec, (; eigvals=[eigval])
end