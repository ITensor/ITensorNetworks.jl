# a large number to prevent this edge being a cut
MAX_WEIGHT = 1e32

"""
GraphsFlows.mincut overload, calculate the mincut between two subsets of the uncontracted inds
(source_inds and terminal_inds) of the input network.
Mincut of two inds list is defined as the mincut of two newly added vertices,
each one neighboring to one inds subset.
"""
function GraphsFlows.mincut(
  network::Vector{ITensor}, source_inds::Vector{<:Index}, terminal_inds::Vector{<:Index}
)
  @assert length(source_inds) >= 1
  @assert length(terminal_inds) >= 1
  noncommon_inds = noncommoninds(network...)
  @assert issubset(source_inds, noncommon_inds)
  @assert issubset(terminal_inds, noncommon_inds)
  tn = ITensorNetwork([network..., ITensor(source_inds...), ITensor(terminal_inds...)])
  return GraphsFlows.mincut(tn, length(network) + 1, length(network) + 2, weights(tn))
end

"""
Calculate the mincut_partitions between two subsets of the uncontracted inds
(source_inds and terminal_inds) of the input network.
"""
function mincut_partitions(
  network::Vector{ITensor}, source_inds::Vector{<:Index}, terminal_inds::Vector{<:Index}
)
  p1, p2, cut = GraphsFlows.mincut(network, source_inds, terminal_inds)
  p1 = [i for i in p1 if i <= length(network)]
  p2 = [i for i in p2 if i <= length(network)]
  return p1, p2
end

"""
Sum of shortest path distances among all outinds.
"""
function distance(network::Vector{ITensor}, outinds::Vector{<:Index})
  @assert length(outinds) >= 1
  @assert issubset(outinds, noncommoninds(network...))
  if length(outinds) == 1
    return 0.0
  end
  new_tensors = [ITensor(i) for i in outinds]
  tn = ITensorNetwork([network..., new_tensors...])
  distances = 0.0
  for i in (length(network) + 1):(length(network) + length(outinds) - 1)
    ds = dijkstra_shortest_paths(tn, i, weights(tn))
    for j in (i + 1):(length(network) + length(outinds))
      distances += ds.dists[j]
    end
  end
  return distances
end

"""
create a tn with empty ITensors whose outinds weights are MAX_WEIGHT
The maxweight_network is constructed so that only commoninds of the network
will be considered in mincut.
"""
function _maxweightoutinds_network(
  network::Vector{ITensor}, outinds::Union{Nothing,Vector{<:Index}}
)
  @assert issubset(outinds, noncommoninds(network...))
  out_to_maxweight_ind = Dict{Index,Index}()
  for ind in outinds
    out_to_maxweight_ind[ind] = Index(MAX_WEIGHT, ind.tags)
  end
  maxweight_network = Vector{ITensor}()
  for t in network
    inds1 = [i for i in inds(t) if !(i in outinds)]
    inds2 = [out_to_maxweight_ind[i] for i in inds(t) if i in outinds]
    newt = ITensor(inds1..., inds2...)
    push!(maxweight_network, newt)
  end
  return maxweight_network, out_to_maxweight_ind
end

"""
Given a network and outinds (a subset of noncommoninds of network),
get a binary tree structure of outinds based on given algorithm (mincut or mps).
If algorithm="mps", the binary tree will have a line/mps structure.

The binary tree is recursively constructed from leaves to the root.

Example:
# TODO
"""
function inds_binary_tree(
  network::Vector{ITensor}, outinds::Union{Nothing,Vector{<:Index}}; algorithm="mincut"
)
  if outinds == nothing
    outinds = noncommoninds(network...)
  end
  if length(outinds) == 1
    return outinds
  end
  maxweight_network, out_to_maxweight_ind = _maxweightoutinds_network(network, outinds)
  return _inds_binary_tree(
    network => maxweight_network, out_to_maxweight_ind; algorithm=algorithm
  )
end

function _inds_binary_tree(
  network_pair::Pair{Vector{ITensor},Vector{ITensor}},
  out_to_maxweight_ind::Dict{Index,Index};
  algorithm="mincut",
)
  @assert algorithm in ["mincut", "mps"]
  if algorithm == "mincut"
    return _inds_binary_tree_mincut(network_pair, out_to_maxweight_ind)
  elseif algorithm == "mps"
    return line_to_tree(_inds_mps_order(network_pair, out_to_maxweight_ind))
  end
end

"""
Given a network and outinds, returns a vector of indices representing MPS inds ordering.
"""
function inds_mps_order(network::Vector{ITensor}, outinds::Union{Nothing,Vector{<:Index}})
  if outinds == nothing
    outinds = noncommoninds(network...)
  end
  if length(outinds) == 1
    return outinds
  end
  p_network, out_to_maxweight_ind = _maxweightoutinds_network(network, outinds)
  return _inds_mps_order(network => p_network, out_to_maxweight_ind)
end

function _inds_mps_order(
  network_pair::Pair{Vector{ITensor},Vector{ITensor}},
  out_to_maxweight_ind::Dict{Index,Index},
)
  outinds = collect(keys(out_to_maxweight_ind))
  @assert length(outinds) >= 1
  if length(outinds) <= 2
    return outinds
  end
  first_inds, _ = _mincut_inds(
    network_pair, out_to_maxweight_ind, collect(powerset(outinds, 1, 1))
  )
  first_ind = first_inds[1]
  linear_order = [first_ind]
  outinds = setdiff(outinds, linear_order)
  while length(outinds) > 1
    sourceinds_list = [Vector{Index}([linear_order..., i]) for i in outinds]
    target_inds, _ = _mincut_inds(network_pair, out_to_maxweight_ind, sourceinds_list)
    new_ind = setdiff(target_inds, linear_order)[1]
    push!(linear_order, new_ind)
    outinds = setdiff(outinds, [new_ind])
  end
  push!(linear_order, outinds[1])
  return linear_order
end

function _inds_binary_tree_mincut(
  network_pair::Pair{Vector{ITensor},Vector{ITensor}},
  out_to_maxweight_ind::Dict{Index,Index},
)
  outinds = collect(keys(out_to_maxweight_ind))
  @assert length(outinds) >= 1
  if length(outinds) <= 2
    return outinds
  end
  while length(outinds) > 2
    tree_list = collect(powerset(outinds, 2, 2))
    sourceinds_list = [vectorize(i) for i in tree_list]
    _, i = _mincut_inds(network_pair, out_to_maxweight_ind, sourceinds_list)
    tree = tree_list[i]
    outinds = setdiff(outinds, tree)
    outinds = vcat([tree], outinds)
  end
  return outinds
end

"""
Find a vector of indices within sourceinds_list yielding the mincut of given network_pair.
Args:
  network_pair: a pair of networks (tn1 => tn2), where tn2 is generated via _maxweightoutinds_network(tn1)
  out_to_maxweight_ind: a dict mapping each out ind in tn1 to out ind in tn2
  sourceinds_list: a list of vector of indices to be considered
Note:
  For each sourceinds in sourceinds_list, we consider its mincut within both networks (tn1, tn2) given in network_pair.
  The mincut in tn1 represents the rank upper bound when splitting sourceinds with other inds in outinds.
  The mincut in tn2 represents the rank upper bound when the weights of outinds are very large.
  The first mincut upper_bounds the number of non-zero singular values, while the second empirically reveals the
  singular value decay.
  We output the sourceinds where the first mincut value is the minimum, the secound mincut value is also
  the minimum under the condition that the first mincut is optimal, and the sourceinds have the lowest all-pair shortest path.
"""
function _mincut_inds(
  network_pair::Pair{Vector{ITensor},Vector{ITensor}},
  out_to_maxweight_ind::Dict{Index,Index},
  sourceinds_list::Vector{<:Vector{<:Index}},
)
  function _mincut_value(network, sinds, outinds)
    tinds = setdiff(outinds, sinds)
    _, _, cut = GraphsFlows.mincut(network, sinds, tinds)
    return cut
  end
  function _get_weights(source_inds, outinds, maxweight_source_inds, maxweight_outinds)
    mincut_val = _mincut_value(network_pair.first, source_inds, outinds)
    maxweight_mincut_val = _mincut_value(
      network_pair.second, maxweight_source_inds, maxweight_outinds
    )
    dist = distance(network_pair.first, source_inds)
    return (mincut_val, maxweight_mincut_val, dist)
  end

  outinds = collect(keys(out_to_maxweight_ind))
  maxweight_outinds = collect(values(out_to_maxweight_ind))
  weights = []
  for source_inds in sourceinds_list
    maxweight_source_inds = [out_to_maxweight_ind[i] for i in source_inds]
    push!(
      weights, _get_weights(source_inds, outinds, maxweight_source_inds, maxweight_outinds)
    )
  end
  i = argmin(weights)
  return sourceinds_list[i], i
end
