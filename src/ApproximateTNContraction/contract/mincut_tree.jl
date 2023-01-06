using ITensorNetworks: ITensorNetwork

# a large number to prevent this edge being a cut
MAX_WEIGHT = 1e32

function _build_pseudo_network(
  network::Vector{ITensor}, outinds::Union{Nothing,Vector{<:Index}}
)
  # create a pseudo_tn (a tn without any data inside) whose outinds weights are MAX_WEIGHT
  out_to_pseudo_ind = Dict{Index,Index}()
  for ind in outinds
    out_to_pseudo_ind[ind] = Index(MAX_WEIGHT, ind.tags)
  end
  pseudo_network = Vector{ITensor}()
  for t in network
    inds1 = [i for i in inds(t) if !(i in outinds)]
    inds2 = [out_to_pseudo_ind[i] for i in inds(t) if i in outinds]
    newt = ITensor(inds1..., inds2...)
    push!(pseudo_network, newt)
  end
  return pseudo_network, out_to_pseudo_ind
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
  p_network, out_to_pseudo_ind = _build_pseudo_network(network, outinds)
  return _inds_binary_tree(network => p_network, out_to_pseudo_ind; algorithm=algorithm)
end

function _inds_binary_tree(
  network_pair::Pair{Vector{ITensor},Vector{ITensor}},
  out_to_pseudo_ind::Dict{Index,Index};
  algorithm="mincut",
)
  @assert algorithm in ["mincut", "mps"]
  if algorithm == "mincut"
    return _mincut_inds(network_pair, out_to_pseudo_ind)
  elseif algorithm == "mps"
    return line_to_tree(_inds_linear_order(network_pair, out_to_pseudo_ind))
  end
end

function inds_linear_order(
  network::Vector{ITensor}, outinds::Union{Nothing,Vector{<:Index}}
)
  if outinds == nothing
    outinds = noncommoninds(network...)
  end
  if length(outinds) == 1
    return outinds
  end
  p_network, out_to_pseudo_ind = _build_pseudo_network(network, outinds)
  return _inds_linear_order(network => p_network, out_to_pseudo_ind)
end

function _inds_linear_order(
  network_pair::Pair{Vector{ITensor},Vector{ITensor}}, out_to_pseudo_ind::Dict{Index,Index}
)
  outinds = collect(keys(out_to_pseudo_ind))
  outinds = [[i] for i in outinds]
  @assert length(outinds) >= 1
  if length(outinds) <= 2
    return outinds
  end
  new_edge = new_edge_mincut(
    network_pair, out_to_pseudo_ind, collect(powerset(outinds, 1, 1))
  )
  new_edge = new_edge[1]
  linear_order = [new_edge]
  while length(outinds) > 2
    splitinds = [[new_edge, i] for i in outinds if i != new_edge]
    new_edge = new_edge_mincut(network_pair, out_to_pseudo_ind, splitinds)
    push!(linear_order, new_edge[2])
    outinds = setdiff(outinds, new_edge)
    outinds = vcat([new_edge], outinds)
  end
  last_index = [i for i in outinds if i != new_edge]
  @assert length(last_index) == 1
  push!(linear_order, last_index[1])
  return linear_order
end

function _mincut_inds(
  network_pair::Pair{Vector{ITensor},Vector{ITensor}}, out_to_pseudo_ind::Dict{Index,Index}
)
  outinds = collect(keys(out_to_pseudo_ind))
  outinds = [[i] for i in outinds]
  return __mincut_inds(network_pair, out_to_pseudo_ind, outinds)
end

function __mincut_inds(
  network_pair::Pair{Vector{ITensor},Vector{ITensor}},
  out_to_pseudo_ind::Dict{Index,Index},
  outinds::Vector{<:Vector},
)
  @assert length(outinds) >= 1
  if length(outinds) == 1
    return outinds[1]
  end
  if length(outinds) == 2
    return outinds
  end
  new_edge = new_edge_mincut(
    network_pair, out_to_pseudo_ind, collect(powerset(outinds, 2, 2))
  )
  outinds = setdiff(outinds, new_edge)
  outinds = vcat([new_edge], outinds)
  return __mincut_inds(network_pair, out_to_pseudo_ind, outinds)
end

function new_edge_mincut(
  network_pair::Pair{Vector{ITensor},Vector{ITensor}},
  out_to_pseudo_ind::Dict{Index,Index},
  split_inds_list::Vector,
)
  outinds = collect(keys(out_to_pseudo_ind))
  p_outinds = collect(values(out_to_pseudo_ind))
  mincuts, pseudo_mincuts, dists = [], [], []
  for split_inds in split_inds_list
    if length(split_inds) == 2
      push!(
        dists,
        distance(network_pair.first, vectorize(split_inds[1]), vectorize(split_inds[2])),
      )
    else
      push!(dists, 0.0)
    end
    split_inds = vectorize(split_inds)
    p_split_inds = [out_to_pseudo_ind[i] for i in split_inds]
    push!(mincuts, mincut_value(network_pair.first, split_inds, outinds)[3])
    push!(pseudo_mincuts, mincut_value(network_pair.second, p_split_inds, p_outinds)[3])
  end
  indices_min = [i for i in 1:length(mincuts) if mincuts[i] == min(mincuts...)]
  selected_pseudo_mincuts = [pseudo_mincuts[i] for i in indices_min]
  indices_min = [
    i for i in indices_min if pseudo_mincuts[i] == min(selected_pseudo_mincuts...)
  ]
  dists_min = [dists[i] for i in indices_min]
  _, index = findmin(dists_min)
  i = indices_min[index]
  new_edge = split_inds_list[i]
  return new_edge
end

function distance(network::Vector{ITensor}, inds1::Vector{<:Index}, inds2::Vector{<:Index})
  new_t1 = ITensor(inds1...)
  new_t2 = ITensor(inds2...)
  tn = ITensorNetwork([network..., new_t1, new_t2])
  ds = dijkstra_shortest_paths(tn, length(network) + 1, weights(tn))
  return ds.dists[length(network) + 2]
end

function mincut_value(
  network::Vector{ITensor}, source_inds::Vector{<:Index}, out_inds::Vector{<:Index}
)
  terminal_inds = setdiff(out_inds, source_inds)
  new_t1 = ITensor(source_inds...)
  new_t2 = ITensor(terminal_inds...)
  tn = ITensorNetwork([network..., new_t1, new_t2])
  return GraphsFlows.mincut(tn, length(network) + 1, length(network) + 2, weights(tn))
end

function mincut_subnetwork(
  network::Vector{ITensor}, source_inds::Vector{<:Index}, out_inds::Vector{<:Index}
)
  @timeit timer "mincut_subnetwork" begin
    if length(source_inds) == length(out_inds)
      return network
    end
    p_network, out_to_pseudo_ind = _build_pseudo_network(network, out_inds)
    p_source_inds = [out_to_pseudo_ind[i] for i in source_inds]
    p_out_inds = [out_to_pseudo_ind[i] for i in out_inds]
    part1, part2, val = mincut_value(p_network, p_source_inds, p_out_inds)
    @assert length(part1) > 1
    @assert length(part2) > 1
    return [network[i] for i in part1 if i <= length(network)]
  end
end
