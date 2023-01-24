include("union_find.jl")

function tree_embedding(network::Vector{OrthogonalITensor}, inds_btree::Vector; algorithm)
  # TODO: consider identity matrices
  tensor_to_ortho_tensor = Dict{ITensor,OrthogonalITensor}()
  for ortho_tensor in network
    tensor_to_ortho_tensor[ortho_tensor.tensor] = ortho_tensor
  end
  tnets_dict = tree_embedding(get_tensors(network), inds_btree; algorithm=algorithm)
  ortho_tnets_dict = Dict()
  for (key, tensors) in tnets_dict
    ortho_tensors = Vector{OrthogonalITensor}()
    for t in tensors
      if haskey(tensor_to_ortho_tensor, t)
        push!(ortho_tensors, tensor_to_ortho_tensor[t])
      else
        push!(ortho_tensors, OrthogonalITensor(t))
      end
    end
    ortho_tnets_dict[key] = ortho_tensors
  end
  return ortho_tnets_dict
end

function mincut_subnetwork_insert_deltas(
  network::Vector{ITensor}, source_inds::Vector{<:Index}
)
  out_inds = noncommoninds(network...)
  # terminal_inds = setdiff(out_inds, source_inds)
  # tensors_to_add_delta = []
  # for t in network
  #   t_inds = inds(t)
  #   if length(intersect(source_inds, t_inds)) > 0 && length(intersect(terminal_inds, t_inds)) > 0
  #     push!(tensors_to_add_delta, t)
  #   end
  # end
  # inds_to_add_delta = []
  # for t in tensors_to_add_delta
  #   uncontract_inds = intersect(inds(t), out_inds)
  #   inds_to_add_delta = [inds_to_add_delta..., uncontract_inds...]
  # end
  # deltas, networkprime, _ = split_deltas(inds_to_add_delta, network)
  deltas, networkprime, _ = split_deltas(noncommoninds(network...), network)
  network = Vector{ITensor}(vcat(deltas, networkprime))
  source_subnetwork = mincut_subnetwork(network, source_inds, out_inds)
  remain_network = setdiff(network, source_subnetwork)
  source_subnetwork = simplify_deltas(source_subnetwork)
  remain_network = simplify_deltas(remain_network)
  @assert (
    length(noncommoninds(network...)) ==
    length(noncommoninds(source_subnetwork..., remain_network...))
  )
  return source_subnetwork, remain_network
end

function tree_embedding(network::Vector{ITensor}, inds_btree::Vector; algorithm)
  btree_to_output_tn = Dict{Vector,Vector{ITensor}}()
  btree_to_input_tn = Dict{Vector,Vector{ITensor}}()
  btree_to_input_tn[inds_btree] = network
  nodes = reverse(topo_sort(inds_btree; type=Vector{<:Vector}))
  nodes = [nodes..., get_leaves(inds_btree)...]
  for node in nodes
    @assert haskey(btree_to_input_tn, node)
    input_tn = btree_to_input_tn[node]
    # @info "node", node
    if length(node) == 1
      btree_to_output_tn[node] = input_tn
      continue
    end
    net1, input_tn = mincut_subnetwork_insert_deltas(input_tn, vectorize(node[1]))
    btree_to_input_tn[node[1]] = net1
    net1, input_tn = mincut_subnetwork_insert_deltas(input_tn, vectorize(node[2]))
    btree_to_input_tn[node[2]] = net1
    btree_to_output_tn[node] = input_tn
    # @info "btree_to_output_tn[node]", btree_to_output_tn[node]
  end
  if algorithm == "svd"
    return btree_to_output_tn
  else
    return remove_deltas(btree_to_output_tn)
  end
end

is_delta(t) = (t.tensor.storage.data == 1.0)

function simplify_deltas(network::Vector{ITensor})
  out_delta_inds = Vector{Pair}()
  # outinds will always be the roots in union-find
  outinds = noncommoninds(network...)
  deltas = filter(t -> is_delta(t), network)
  inds_list = map(t -> collect(inds(t)), deltas)
  deltainds = collect(Set(vcat(inds_list...)))
  uf = UF(deltainds)
  for t in deltas
    i1, i2 = inds(t)
    if root(uf, i1) in outinds && root(uf, i2) in outinds
      push!(out_delta_inds, root(uf, i1) => root(uf, i2))
    end
    if root(uf, i1) in outinds
      connect(uf, i2, i1)
    else
      connect(uf, i1, i2)
    end
  end
  sim_dict = Dict([ind => root(uf, ind) for ind in deltainds])
  network = setdiff(network, deltas)
  network = replaceinds(network, sim_dict)
  out_delta = [delta(i.first, i.second) for i in out_delta_inds]
  return Vector{ITensor}([network..., out_delta...])
end

# remove deltas to improve the performance
function remove_deltas(tnets_dict::Dict)
  # only remove deltas in intermediate nodes
  ks = filter(k -> (length(k) > 1), collect(keys(tnets_dict)))
  network = vcat([tnets_dict[k] for k in ks]...)
  # outinds will always be the roots in union-find
  outinds = noncommoninds(network...)

  deltas = filter(t -> is_delta(t), network)
  inds_list = map(t -> collect(inds(t)), deltas)
  deltainds = collect(Set(vcat(inds_list...)))
  uf = UF(deltainds)
  for t in deltas
    i1, i2 = inds(t)
    if root(uf, i1) in outinds
      connect(uf, i2, i1)
    else
      connect(uf, i1, i2)
    end
  end
  sim_dict = Dict([ind => root(uf, ind) for ind in deltainds])
  for k in ks
    net = tnets_dict[k]
    net = setdiff(net, deltas)
    tnets_dict[k] = replaceinds(net, sim_dict)
    # @info "$(k), $(TreeTensor(net...))"
  end
  return tnets_dict
end

function split_deltas(inds, subnet)
  sim_dict = Dict([ind => sim(ind) for ind in inds])
  deltas = [delta(i, sim_dict[i]) for i in inds]
  subnet = replaceinds(subnet, sim_dict)
  return deltas, subnet, collect(values(sim_dict))
end
