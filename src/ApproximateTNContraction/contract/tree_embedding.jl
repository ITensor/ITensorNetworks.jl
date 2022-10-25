include("union_find.jl")

function tree_embedding(network::Vector{OrthogonalITensor}, inds_btree::Vector)
  # TODO: consider identity matrices
  tensor_to_ortho_tensor = Dict{ITensor,OrthogonalITensor}()
  for ortho_tensor in network
    tensor_to_ortho_tensor[ortho_tensor.tensor] = ortho_tensor
  end
  tnets_dict = tree_embedding(get_tensors(network), inds_btree)
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

function tree_embedding(network::Vector{ITensor}, inds_btree::Vector)
  deltas, networkprime, _ = split_deltas(noncommoninds(network...), network)
  network = vcat(deltas, networkprime)
  # tnets_dict map each inds_btree node to a tensor network
  tnets_dict = Dict()
  function embed(tree::Vector)
    if length(tree) == 1
      # add delta to handle the case with two output edges neighboring to one tensor
      # being split (MPS case with element grouping)
      ind = tree[1]
      sim_dict = Dict([ind => sim(ind)])
      tnets_dict[tree] = [delta(ind, sim_dict[ind])]
      network = replaceinds(network, sim_dict)
      return Tuple([sim_dict[ind]])
    end
    ind1 = embed(tree[1])
    ind2 = embed(tree[2])
    deltas, splitinds, tnets_dict[tree[1]] = insert_deltas(ind1, ind2, tnets_dict[tree[1]])
    network = Vector{ITensor}(vcat(network, deltas))
    # use mincut to get the subnetwork
    subnetwork = mincut_subnetwork(network, splitinds, noncommoninds(network...))
    subsplitinds = intersect(splitinds, noncommoninds(subnetwork...))
    remaininds = collect(setdiff(noncommoninds(subnetwork...), subsplitinds))
    network = collect(setdiff(network, subnetwork))
    # remaininds
    deltas, subnetwork, _ = split_deltas(remaininds, subnetwork)
    network = vcat(network, deltas)
    # subsplitinds
    inds = collect(setdiff(splitinds, subsplitinds))
    if length(inds) > 0
      inds = Vector{Index}(inds)
      deltas, network, _ = split_deltas(inds, network)
      subnetwork = vcat(subnetwork, deltas)
    end
    # @info "$(tree), $(TreeTensor(subnetwork...))"
    tnets_dict[tree] = subnetwork
    return Tuple(setdiff(noncommoninds(subnetwork...), splitinds))
  end
  @assert (length(inds_btree) >= 2)
  embed(inds_btree)
  return remove_deltas(tnets_dict)
end

is_delta(t) = (t.tensor.storage.data == 1.0)

# remove deltas to improve the performance
function remove_deltas(tnets_dict)
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

function insert_deltas(ind1, ind2, subnet1)
  intersect_inds = intersect(ind1, ind2)
  ind1_unique = collect(setdiff(ind1, intersect_inds))
  ind2_unique = collect(setdiff(ind2, intersect_inds))
  outinds = vcat(ind1_unique, ind2_unique)
  # look at intersect_inds
  deltas = []
  if length(intersect_inds) >= 1
    deltas, subnet1, siminds = split_deltas(intersect_inds, subnet1)
    outinds = vcat(outinds, intersect_inds, siminds)
  end
  return deltas, outinds, subnet1
end
