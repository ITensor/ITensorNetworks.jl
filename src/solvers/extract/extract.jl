# Here extract_local_tensor and insert_local_tensor
# are essentially inverse operations, adapted for different kinds of 
# algorithms and networks.
#
# In the simplest case, exact_local_tensor contracts together a few
# tensors of the network and returns the result, while 
# insert_local_tensors takes that tensor and factorizes it back
# apart and puts it back into the network.
#
function default_extracter(state, projected_operator, region; internal_kwargs)
  if isa(region, AbstractEdge)
    vsrc, vdst = src(region), dst(region)
    state = orthogonalize(state, vsrc)
    left_inds = uniqueinds(state[vsrc], state[vdst])
    #ToDo: replace with call to factorize
    U, S, V = svd(
      state[vsrc], left_inds; lefttags=tags(state, region), righttags=tags(state, region)
    )
    state[vsrc] = U
    local_tensor = S * V
  else
    state = orthogonalize(state, region)
    local_tensor = prod(state[v] for v in region)
  end
  projected_operator = position(projected_operator, state, region)
  return state, projected_operator, local_tensor
end
