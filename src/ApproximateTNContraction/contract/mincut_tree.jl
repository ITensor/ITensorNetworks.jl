
# a large number to prevent this edge being a cut
MAX_WEIGHT = 100000

function inds_binary_tree(
  network::Vector{ITensor}, inds_groups::Vector{<:Vector}; kwargs...
)
  tng = TensorNetworkGraph(network, vectorize(inds_groups))
  function get_sub_tree(inds)
    @assert all(ind -> ind isa Index, inds)
    if length(inds) == 1
      return inds
    end
    inds = [[i] for i in inds]
    return inds_binary_tree!(tng, inds; kwargs...)
  end
  inds_groups = [get_sub_tree(inds) for inds in inds_groups]
  if length(inds_groups) <= 2
    return inds_groups
  end
  return inds_binary_tree!(tng, inds_groups; kwargs...)
end

function inds_binary_tree!(tng::TensorNetworkGraph, outinds::Vector; algorithm="mincut")
  @assert algorithm in ["mincut", "mincut-mps", "mps"]
  @assert all(ind -> ind in keys(tng.out_edge_dict), outinds)
  if algorithm == "mincut"
    return mincut_inds!(tng, outinds)
  elseif algorithm == "mincut-mps"
    inds_tree = mincut_inds!(tng, outinds)
    linear_tree = linearize(inds_tree, tng)
    out_inds = linear_tree[1]
    for i in 2:length(linear_tree)
      out_inds = [out_inds, linear_tree[i]]
    end
    return out_inds
  elseif algorithm == "mps"
    return mps_inds!(tng, outinds)
  end
end

function inds_binary_tree(
  network::Vector{ITensor}, outinds::Union{Nothing,Vector{<:Index}}; algorithm="mincut"
)
  if outinds == nothing
    outinds = noncommoninds(network...)
  end
  if length(outinds) == 1
    return outinds
  end
  if algorithm == "sequential-mps"
    out_inds = [outinds[1]]
    for i in 2:length(outinds)
      out_inds = [out_inds, [outinds[i]]]
    end
    return out_inds
  end
  tng = TensorNetworkGraph(network, outinds)
  grouped_uncontracted_inds = [[i] for i in outinds]
  return inds_binary_tree!(tng, grouped_uncontracted_inds; algorithm=algorithm)
end

#TODO: rewrite this function
#TODO: pick one end deterministically
function linearize(inds_tree::Vector, tng::TensorNetworkGraph)
  get_dist(edge, distances) = distances.dists[tng.inner_edge_dict[edge][1]]
  function get_boundary_dists(line, source)
    first, last = line[1], line[end]
    ds = dijkstra_shortest_paths(tng.graph, source, tng.weights)
    return get_dist(first, ds), get_dist(last, ds)
  end

  if length(inds_tree) == 1
    return inds_tree
  end
  left = linearize(inds_tree[1], tng)
  right = linearize(inds_tree[2], tng)
  if length(left) == 1 && length(right) == 1
    return [left, right]
  end
  if length(left) == 1
    source = tng.inner_edge_dict[left][1]
    dist_first, dist_last = get_boundary_dists(right, source)
    if dist_last < dist_first
      right = reverse(right)
    end
    return [left, right...]
  end
  if length(right) == 1
    source = tng.inner_edge_dict[right][1]
    dist_first, dist_last = get_boundary_dists(left, source)
    if dist_last > dist_first
      left = reverse(left)
    end
    return [left..., right]
  end
  s1, s2 = tng.inner_edge_dict[left[1]][1], tng.inner_edge_dict[left[end]][1]
  dist1_first, dist1_last = get_boundary_dists(right, s1)
  dist2_first, dist2_last = get_boundary_dists(right, s2)
  if min(dist1_first, dist1_last) < min(dist2_first, dist2_last)
    left = reverse(left)
    if dist1_last < dist1_first
      right = reverse(right)
    end
  else
    if dist2_last < dist2_first
      right = reverse(right)
    end
  end
  return [left..., right...]
end

function mincut_subnetwork(
  network::Vector{ITensor}, sourceinds::Vector, uncontract_inds::Vector
)
  @timeit timer "mincut_subnetwork" begin
    if length(sourceinds) == length(uncontract_inds)
      return network
    end
    tng = TensorNetworkGraph(network)
    grouped_sourceinds = [[ind] for ind in sourceinds]
    part1, part2, mincut = mincut_value(tng, grouped_sourceinds)
    @assert length(part1) > 1
    @assert length(part2) > 1
    return [network[i] for i in part1 if i <= length(network)]
  end
end

function mincut_inds!(tng::TensorNetworkGraph, outinds::Vector)
  @assert length(outinds) >= 1
  # base case here, for the case length(outinds) == 2, we still need to do the update
  if length(outinds) == 1
    return outinds[1]
  end
  new_edge, minval = new_edge_mincut(tng, collect(powerset(outinds, 2, 2)))
  outinds = update!(tng, outinds, new_edge, minval)
  return mincut_inds!(tng, outinds)
end

function mincut_inds(tng::TensorNetworkGraph, uncontract_inds::Vector)
  @timeit timer "mincut_inds" begin
    tng = copy(tng)
    uncontract_inds = copy(uncontract_inds)
    return mincut_inds!(tng, uncontract_inds)
  end
end

function mps_inds!(tng::TensorNetworkGraph, outinds::Vector)
  @assert length(outinds) >= 1
  # base case here, for the case length(outinds) == 2, we still need to do the update
  if length(outinds) == 1
    return outinds[1]
  end
  new_edge, minval = new_edge_mincut(tng, collect(powerset(outinds, 2, 2)))
  outinds = update!(tng, outinds, new_edge, minval)
  first_ind = new_edge
  while length(outinds) > 2
    splitinds = [[first_ind, i] for i in outinds if i != first_ind]
    new_edge, minval = new_edge_mincut(tng, splitinds)
    outinds = update!(tng, outinds, new_edge, minval)
    first_ind = new_edge
  end
  return outinds
end

function mps_inds(tng::TensorNetworkGraph, uncontract_inds::Vector)
  tng = copy(tng)
  uncontract_inds = copy(uncontract_inds)
  return mps_inds!(tng, uncontract_inds)
end

# update the graph
function update!(tng::TensorNetworkGraph, uncontract_inds::Vector, new_edge::Vector, minval)
  add_vertex!(tng.graph)
  last_vertex = size(tng.graph)[1]
  u1, w_u1 = tng.out_edge_dict[new_edge[1]]
  u2, w_u2 = tng.out_edge_dict[new_edge[2]]
  Graphs.add_edge!(tng.graph, u1, last_vertex)
  Graphs.add_edge!(tng.graph, u2, last_vertex)
  Graphs.add_edge!(tng.graph, last_vertex, u1)
  Graphs.add_edge!(tng.graph, last_vertex, u2)
  new_weights = zeros(last_vertex, last_vertex)
  new_weights[1:(last_vertex - 1), 1:(last_vertex - 1)] = tng.weights
  #if not setting to MAX_WEIGHT would affect later tree selections
  new_weights[u1, last_vertex] = MAX_WEIGHT
  new_weights[u2, last_vertex] = MAX_WEIGHT
  new_weights[last_vertex, u1] = MAX_WEIGHT
  new_weights[last_vertex, u2] = MAX_WEIGHT
  # update the dict
  tng.inner_edge_dict[new_edge[1]] = (u1, last_vertex, MAX_WEIGHT)#w_u1)
  tng.inner_edge_dict[new_edge[2]] = (u2, last_vertex, MAX_WEIGHT)#w_u2)
  delete!(tng.out_edge_dict, new_edge[1])
  delete!(tng.out_edge_dict, new_edge[2])
  tng.out_edge_dict[new_edge] = (last_vertex, minval)
  # update uncontract_inds
  uncontract_inds = setdiff(uncontract_inds, new_edge)
  uncontract_inds = vcat([new_edge], uncontract_inds)
  tng.weights = new_weights
  return uncontract_inds
end

# TODO: rewrite this function
function new_edge_mincut(tng::TensorNetworkGraph, split_inds_list::Vector)
  mincuts = [mincut_value(tng, split_inds)[3] for split_inds in split_inds_list]
  split_sizes = [
    sum([tng.out_edge_dict[ind][2] for ind in split_inds]) for split_inds in split_inds_list
  ]
  dists = [distance(tng, inds...) for inds in split_inds_list]
  weights = [min(mincuts[i], split_sizes[i]) for i in 1:length(mincuts)]
  indices_min = [i for i in 1:length(mincuts) if weights[i] == min(weights...)]
  cuts_min = [mincuts[i] for i in indices_min]
  indices_min = [i for i in indices_min if mincuts[i] == min(cuts_min...)]
  dists_min = [dists[i] for i in indices_min]
  _, index = findmin(dists_min)
  i = indices_min[index]
  minval = weights[i]
  new_edge = split_inds_list[i]
  return new_edge, minval
end

function mincut_value(tng::TensorNetworkGraph, split_inds::Vector)
  tng = copy(tng)
  # add two vertices to the graph to model the s and t
  add_vertices!(tng.graph, 2)
  t = size(tng.graph)[1]
  s = t - 1
  new_weights = zeros(t, t)
  new_weights[1:(t - 2), 1:(t - 2)] = tng.weights
  for ind in split_inds
    u, _ = tng.out_edge_dict[ind]
    Graphs.add_edge!(tng.graph, u, s)
    Graphs.add_edge!(tng.graph, s, u)
    new_weights[u, s] = MAX_WEIGHT
    new_weights[s, u] = MAX_WEIGHT
  end
  terminal_inds = setdiff(noncommoninds(tng), split_inds)
  for ind in terminal_inds
    u, _ = tng.out_edge_dict[ind]
    Graphs.add_edge!(tng.graph, u, t)
    Graphs.add_edge!(tng.graph, t, u)
    new_weights[u, t] = MAX_WEIGHT
    new_weights[t, u] = MAX_WEIGHT
  end
  # this t and s sequence makes sure part1 is the largest subgraph yielding mincut
  part2, part1, flow = GraphsFlows.mincut(
    tng.graph, t, s, new_weights, EdmondsKarpAlgorithm()
  )
  return part1, part2, flow
end
