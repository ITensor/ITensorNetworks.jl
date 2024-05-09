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

function bp_inserter(ψ::AbstractITensorNetwork, ψAψ_bpc::BeliefPropagationCache, 
  ψIψ_bpc::BeliefPropagationCache, state::ITensor, region; cache_update_kwargs = (;), kwargs...)

  spec = nothing

  form_network = unpartitioned_graph(partitioned_tensornetwork(ψAψ_bpc))
  if length(region) == 1
    states = [state]
  elseif length(region) == 2
    v1, v2 = region[1], region[2]
    e = edgetype(ψ)(v1, v2)
    pe = partitionedge(ψAψ_bpc, ket_vertex(form_network, v1) => bra_vertex(form_network, v2))
    stateᵥ₁, stateᵥ₂, spec = factorize_svd(state,uniqueinds(ψ[v1], ψ[v2]); ortho="none", tags=edge_tag(e),kwargs...)
    states = noprime.([stateᵥ₁, stateᵥ₂])
    delete!(messages(ψAψ_bpc), pe)
    delete!(messages(ψIψ_bpc), pe)
    delete!(messages(ψAψ_bpc), reverse(pe))
    delete!(messages(ψIψ_bpc), reverse(pe))
  end

  for (i, v) in enumerate(region)
    state = states[i]
    form_bra_v, form_ket_v = bra_vertex(form_network, v), ket_vertex(form_network, v)
    ψ[v] =state
    state_dag = replaceinds(dag(state), inds(state), dual_index_map(form_network).(inds(state)))
    ψAψ_bpc = update_factor(ψAψ_bpc, form_ket_v, state)
    ψAψ_bpc = update_factor(ψAψ_bpc, form_bra_v, state_dag)
    ψIψ_bpc = update_factor(ψIψ_bpc, form_ket_v, state)
    ψIψ_bpc = update_factor(ψIψ_bpc, form_bra_v, state_dag)
  end

  ψAψ_bpc = update(ψAψ_bpc; cache_update_kwargs...)
  ψIψ_bpc = update(ψIψ_bpc; cache_update_kwargs...)
  updated_ψAψ = unpartitioned_graph(partitioned_tensornetwork(ψAψ_bpc))
  updated_ψIψ = unpartitioned_graph(partitioned_tensornetwork(ψIψ_bpc))
  eigval = scalar(updated_ψAψ; cache! = Ref(ψAψ_bpc), alg = "bp") / scalar(updated_ψIψ; cache! = Ref(ψIψ_bpc), alg = "bp")

  return ψ, ψAψ_bpc, ψIψ_bpc, spec,   (; eigvals=[eigval])
end