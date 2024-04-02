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

function extract_and_truncate(state,projected_operator, region, ortho; maxdim=nothing, cutoff=nothing,internal_kwargs)
  #ToDo: move default into function signature?
  #svd_kwargs = NamedTuple()
  #svd_kwargs = isnothing(maxdim) ? svd_kwargs : merge(svd_kwargs,(;maxdim))
  #svd_kwargs = isnothing(cutoff) ? svd_kwargs : merge(svd_kwargs,(;cutoff))
  svd_kwargs= (;
  (isnothing(maxdim) ? (;) : (;maxdim))...,
  (isnothing(cutoff) ? (;) : (;cutoff))...
  )
  state = orthogonalize(state, ortho)
  if isa(region, AbstractEdge)
    other_vertex = only(setdiff(support(region), [ortho]))
    left_inds = uniqueinds(state[ortho], state[other_vertex])
    #ToDo: replace with call to factorize
    U, S, V = svd(
      state[ortho], left_inds; lefttags=tags(state, region), righttags=tags(state, region), svd_kwargs...
    )
    state[ortho] = U
    local_tensor = S * V
  else
    local_tensor = prod(state[v] for v in region)
  end
  projected_operator = position(projected_operator, state, region)
  return state, projected_operator, local_tensor
end