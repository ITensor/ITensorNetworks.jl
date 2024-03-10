# Here extract_local_tensor and insert_local_tensor
# are essentially inverse operations, adapted for different kinds of 
# algorithms and networks.
#
# In the simplest case, exact_local_tensor contracts together a few
# tensors of the network and returns the result, while 
# insert_local_tensors takes that tensor and factorizes it back
# apart and puts it back into the network.
#
#=
function extract_prolog(state::AbstractTTN, region)
    return state = orthogonalize(state, current_ortho(region))
  end
  
function extract_epilog(state::AbstractTTN, projected_operator, region)
#nsites = (region isa AbstractEdge) ? 0 : length(region)
#projected_operator = set_nsite(projected_operator, nsites) #not necessary
projected_operator = position(projected_operator, state, region)
return projected_operator   #should it return only projected_operator
end
  
  function extract_local_tensor(
    state::AbstractTTN, projected_operator, pos::Vector; extract_kwargs...
  )
    state = extract_prolog(state, pos)
    projected_operator = extract_epilog(state, projected_operator, pos)
    return state, projected_operator, prod(state[v] for v in pos)
  end
  
  function extract_local_tensor(
    state::AbstractTTN, projected_operator, e::AbstractEdge; extract_kwargs...
  )
    state = extract_prolog(state, e)
    left_inds = uniqueinds(state, e)
    #ToDo: do not rely on directionality of edge
    U, S, V = svd(state[src(e)], left_inds; lefttags=tags(state, e), righttags=tags(state, e))
    
    state[src(e)] = U
    projected_operator = extract_epilog(state, projected_operator, e)
    return state, projected_operator, S * V
  end
=#
function extract_local_tensor(state,projected_operator, region, ortho;internal_kwargs)
    state=orthogonalize(state, ortho)
    if isa(region,AbstractEdge)
        other_vertex=only(setdiff(support(region),[ortho]))
        #this is replicating some higher level code that requires directed edge
        #alternatively, use existing logic for edges and revert the edge if it happens to be the wrong way
        left_inds = uniqueinds(state[ortho],state[other_vertex])
        #ToDo: replace with call to factorize
        U, S, V = svd(state[ortho], left_inds; lefttags=tags(state, region), righttags=tags(state, region))
        state[ortho] = U
        local_tensor = S*V
    else
        local_tensor = prod(state[v] for v in region)
    end
    projected_operator = position(projected_operator, state, region)   
    return state, projected_operator, local_tensor
end