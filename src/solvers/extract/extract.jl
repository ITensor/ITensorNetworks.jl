# Here extract_local_tensor and insert_local_tensor
# are essentially inverse operations, adapted for different kinds of 
# algorithms and networks.
#
# In the simplest case, exact_local_tensor contracts together a few
# tensors of the network and returns the result, while 
# insert_local_tensors takes that tensor and factorizes it back
# apart and puts it back into the network.
#
function default_extracter(state, projected_operator, region, ortho; internal_kwargs)
  state = orthogonalize(state, ortho)
  if isa(region, AbstractEdge)
    other_vertex = only(setdiff(support(region), [ortho]))
    left_inds = uniqueinds(state[ortho], state[other_vertex])
    #ToDo: replace with call to factorize
    U, S, V = svd(
      state[ortho], left_inds; lefttags=tags(state, region), righttags=tags(state, region)
    )
    state[ortho] = U
    local_tensor = S * V
  else
    local_tensor = prod(state[v] for v in region)
  end
  projected_operator = position(projected_operator, state, region)
  return state, projected_operator, local_tensor
end

function bp_extracter(ψ::AbstractITensorNetwork, ψAψ_bpc::BeliefPropagationCache, ψIψ_bpc::BeliefPropagationCache, region;
  regularization = 10*eps(scalartype(ψ)))
  form_network = unpartitioned_graph(partitioned_tensornetwork(ψAψ_bpc))
  form_ket_vertices, form_bra_vertices = ket_vertices(form_network, region), bra_vertices(form_network, region)
  ∂ψAψ_bpc_∂r = environment(ψAψ_bpc, [form_ket_vertices; form_bra_vertices])
  state = prod(ψ[v] for v in region)
  messages = environment(ψIψ_bpc, partitionvertices(ψIψ_bpc, form_ket_vertices))
  f_sqrt = sqrt ∘ (x -> x + regularization)
  f_inv_sqrt = inv ∘ sqrt ∘ (x -> x + regularization)
  sqrt_mts = [ITensorsExtensions.map_eigvals(f_sqrt, mt, inds(mt)[1], inds(mt)[2]; ishermitian=true) for mt in messages]
  inv_sqrt_mts = [ITensorsExtensions.map_eigvals(f_inv_sqrt, mt, inds(mt)[1], inds(mt)[2]; ishermitian=true) for mt in messages]

  return state, ∂ψAψ_bpc_∂r, sqrt_mts, inv_sqrt_mts
end