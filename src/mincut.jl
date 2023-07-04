# a large number to prevent this edge being a cut
MAX_WEIGHT = 1e32

"""
Outputs a maximimally unbalanced directed binary tree DataGraph defining the desired graph structure
"""
function path_graph_structure(tn::ITensorNetwork; alg="top_down")
  return path_graph_structure(tn, noncommoninds(Vector{ITensor}(tn)...); alg=alg)
end

"""
Given a `tn` and `outinds` (a subset of noncommoninds of `tn`), outputs a maximimally unbalanced
directed binary tree DataGraph of `outinds` defining the desired graph structure
"""
function path_graph_structure(tn::ITensorNetwork, outinds::Vector{<:Index}; alg="top_down")
  return _binary_tree_structure(Algorithm(alg), tn, outinds; maximally_unbalanced=true)
end

"""
Outputs a directed binary tree DataGraph defining the desired graph structure
"""
function binary_tree_structure(tn::ITensorNetwork; alg="top_down")
  return binary_tree_structure(tn, noncommoninds(Vector{ITensor}(tn)...); alg=alg)
end

"""
Given a `tn` and `outinds` (a subset of noncommoninds of `tn`), outputs a
directed binary tree DataGraph of `outinds` defining the desired graph structure
"""
function binary_tree_structure(tn::ITensorNetwork, outinds::Vector{<:Index}; alg="top_down")
  return _binary_tree_structure(Algorithm(alg), tn, outinds; maximally_unbalanced=false)
end

"""
Calculate the mincut between two subsets of the uncontracted inds
(source_inds and terminal_inds) of the input tn.
Mincut of two inds list is defined as the mincut of two newly added vertices,
each one neighboring to one inds subset.
"""
function _mincut(
  tn::ITensorNetwork, source_inds::Vector{<:Index}, terminal_inds::Vector{<:Index}
)
  @assert length(source_inds) >= 1
  @assert length(terminal_inds) >= 1
  noncommon_inds = noncommoninds(Vector{ITensor}(tn)...)
  @assert issubset(source_inds, noncommon_inds)
  @assert issubset(terminal_inds, noncommon_inds)
  tn = disjoint_union(
    ITensorNetwork([ITensor(source_inds...), ITensor(terminal_inds...)]), tn
  )
  return GraphsFlows.mincut(tn, (1, 1), (2, 1), weights(tn))
end

"""
Calculate the mincut_partitions between two subsets of the uncontracted inds
(source_inds and terminal_inds) of the input tn.
"""
function _mincut_partitions(
  tn::ITensorNetwork, source_inds::Vector{<:Index}, terminal_inds::Vector{<:Index}
)
  p1, p2, cut = _mincut(tn, source_inds, terminal_inds)
  p1 = [v[1] for v in p1 if v[2] == 2]
  p2 = [v[1] for v in p2 if v[2] == 2]
  return p1, p2
end

function _mincut_partition_maxweightoutinds(
  tn::ITensorNetwork, source_inds::Vector{<:Index}, terminal_inds::Vector{<:Index}
)
  tn, out_to_maxweight_ind = _maxweightoutinds_tn(tn, [source_inds..., terminal_inds...])
  source_inds = [out_to_maxweight_ind[i] for i in source_inds]
  terminal_inds = [out_to_maxweight_ind[i] for i in terminal_inds]
  return _mincut_partitions(tn, source_inds, terminal_inds)
end

"""
Sum of shortest path distances among all outinds.
"""
function _distance(tn::ITensorNetwork, outinds::Vector{<:Index})
  @assert length(outinds) >= 1
  @assert issubset(outinds, noncommoninds(Vector{ITensor}(tn)...))
  if length(outinds) == 1
    return 0.0
  end
  new_tensors = [ITensor(i) for i in outinds]
  tn = disjoint_union(ITensorNetwork(new_tensors), tn)
  distances = 0.0
  for i in 1:(length(new_tensors) - 1)
    ds = dijkstra_shortest_paths(tn, [(i, 1)], weights(tn))
    for j in (i + 1):length(new_tensors)
      distances += ds.dists[(j, 1)]
    end
  end
  return distances
end

"""
create a tn with empty ITensors whose outinds weights are MAX_WEIGHT
The maxweight_tn is constructed so that only commoninds of the tn
will be considered in mincut.
"""
function _maxweightoutinds_tn(tn::ITensorNetwork, outinds::Union{Nothing,Vector{<:Index}})
  @assert issubset(outinds, noncommoninds(Vector{ITensor}(tn)...))
  out_to_maxweight_ind = Dict{Index,Index}()
  for ind in outinds
    out_to_maxweight_ind[ind] = Index(MAX_WEIGHT, ind.tags)
  end
  maxweight_tn = copy(tn)
  for v in vertices(maxweight_tn)
    t = maxweight_tn[v]
    inds1 = [i for i in inds(t) if !(i in outinds)]
    inds2 = [out_to_maxweight_ind[i] for i in inds(t) if i in outinds]
    newt = ITensor(inds1..., inds2...)
    maxweight_tn[v] = newt
  end
  return maxweight_tn, out_to_maxweight_ind
end

function _binary_tree_structure(
  alg::Algorithm"top_down",
  tn::ITensorNetwork,
  outinds::Vector{<:Index};
  maximally_unbalanced::Bool=false,
  backend="Metis",
)
  nested_vector = _recursive_bisection(tn, outinds; backend=backend)
  if maximally_unbalanced
    ordering = collect(Leaves(nested_vector))
    nested_vector = line_to_tree(ordering)
  end
  return _nested_vector_to_digraph(nested_vector)
end

function _map_nested_vector(dict::Dict, nested_vector)
  if haskey(dict, nested_vector)
    return dict[nested_vector]
  end
  return map(v -> _map_nested_vector(dict, v), nested_vector)
end

function _recursive_bisection(tn::ITensorNetwork, outinds::Vector{Set}; backend="Metis")
  tn = copy(tn)
  tensor_to_inds = Dict()
  ts = Vector{ITensor}()
  for is in outinds
    new_t = ITensor(collect(is)...)
    push!(ts, new_t)
    tensor_to_inds[new_t] = is
  end
  new_tn = disjoint_union(tn, ITensorNetwork(ts))
  ts_nested_vector = _recursive_bisection(new_tn, ts; backend=backend)
  return _map_nested_vector(tensor_to_inds, ts_nested_vector)
end

function _recursive_bisection(
  tn::ITensorNetwork,
  target_set::Union{Vector{ITensor},Vector{<:Index}};
  backend="Metis",
  left_inds=Set(),
  right_inds=Set(),
)
  if target_set isa Vector{ITensor}
    set = intersect(target_set, Vector{ITensor}(tn))
  else
    set = intersect(target_set, noncommoninds(Vector{ITensor}(tn)...))
  end
  if length(set) <= 1
    return length(set) == 1 ? set[1] : nothing
  end
  g_parts = partition(tn; npartitions=2, backend=backend)
  # order g_parts
  outinds_1 = noncommoninds(Vector{ITensor}(g_parts[1])...)
  outinds_2 = noncommoninds(Vector{ITensor}(g_parts[2])...)
  left_inds_1 = intersect(left_inds, outinds_1)
  left_inds_2 = intersect(left_inds, outinds_2)
  right_inds_1 = intersect(right_inds, outinds_1)
  right_inds_2 = intersect(right_inds, outinds_2)
  if length(left_inds_2) + length(right_inds_1) > length(left_inds_1) + length(right_inds_2)
    g_parts[1], g_parts[2] = g_parts[2], g_parts[1]
  end
  intersect_inds = intersect(outinds_1, outinds_2)
  set1 = _recursive_bisection(
    g_parts[1],
    target_set;
    backend=backend,
    left_inds=left_inds,
    right_inds=union(right_inds, intersect_inds),
  )
  set2 = _recursive_bisection(
    g_parts[2],
    target_set;
    backend=backend,
    left_inds=union(left_inds, intersect_inds),
    right_inds=right_inds,
  )
  if set1 == nothing || set2 == nothing
    return set1 == nothing ? set2 : set1
  end
  return [set1, set2]
end

"""
Given a tn and outinds (a subset of noncommoninds of tn), get a `DataGraph`
with binary tree structure of outinds that will be used in the binary tree partition.
If maximally_unbalanced=true, the binary tree will have a line/mps structure.
The binary tree is recursively constructed from leaves to the root.

Example:
# TODO
"""
function _binary_tree_structure(
  ::Algorithm"bottom_up",
  tn::ITensorNetwork,
  outinds::Vector{<:Index};
  maximally_unbalanced::Bool=false,
)
  inds_tree_vector = _binary_tree_partition_inds(
    tn, outinds; maximally_unbalanced=maximally_unbalanced
  )
  return _nested_vector_to_digraph(inds_tree_vector)
end

function _binary_tree_partition_inds(
  tn::ITensorNetwork, outinds::Vector{<:Index}; maximally_unbalanced::Bool=false
)
  if length(outinds) == 1
    return outinds
  end
  maxweight_tn, out_to_maxweight_ind = _maxweightoutinds_tn(tn, outinds)
  tn_pair = tn => maxweight_tn
  if maximally_unbalanced == false
    return _binary_tree_partition_inds_mincut(tn_pair, out_to_maxweight_ind)
  else
    return line_to_tree(
      _binary_tree_partition_inds_order_maximally_unbalanced(tn_pair, out_to_maxweight_ind)
    )
  end
end

function _nested_vector_to_digraph(nested_vector::Vector)
  if length(nested_vector) == 1 && !(nested_vector[1] isa Vector)
    inds_btree = DataGraph(NamedDiGraph([1]), Any)
    inds_btree[1] = nested_vector[1]
    return inds_btree
  end
  treenode_to_v = Dict{Union{Vector,Index},Int}()
  graph = DataGraph(NamedDiGraph(), Any)
  v = 1
  for treenode in PostOrderDFS(nested_vector)
    add_vertex!(graph, v)
    treenode_to_v[treenode] = v
    if !(treenode isa Vector)
      graph[v] = treenode
    else
      @assert length(treenode) == 2
      add_edge!(graph, v, treenode_to_v[treenode[1]])
      add_edge!(graph, v, treenode_to_v[treenode[2]])
    end
    v += 1
  end
  return graph
end

"""
Given a tn and outinds, returns a vector of indices representing MPS inds ordering.
"""
function _mps_partition_inds_order(
  tn::ITensorNetwork,
  outinds::Union{Nothing,Vector{<:Index},Vector{Set}};
  alg="top_down",
  backend="Metis",
)
  @assert alg in ["top_down", "bottom_up"]
  if outinds == nothing
    outinds = noncommoninds(Vector{ITensor}(tn)...)
  end
  if length(outinds) == 1
    return outinds
  end
  if alg == "bottom_up"
    tn2, out_to_maxweight_ind = _maxweightoutinds_tn(tn, outinds)
    return _binary_tree_partition_inds_order_maximally_unbalanced(
      tn => tn2, out_to_maxweight_ind
    )
  else
    nested_vector = _recursive_bisection(tn, outinds; backend=backend)
    return filter(v -> v in outinds, collect(PreOrderDFS(nested_vector)))
  end
end

function _binary_tree_partition_inds_order_maximally_unbalanced(
  tn_pair::Pair{<:ITensorNetwork,<:ITensorNetwork}, out_to_maxweight_ind::Dict{Index,Index}
)
  outinds = collect(keys(out_to_maxweight_ind))
  @assert length(outinds) >= 1
  if length(outinds) <= 2
    return outinds
  end
  first_inds, _ = _mincut_inds(
    tn_pair, out_to_maxweight_ind, collect(powerset(outinds, 1, 1))
  )
  first_ind = first_inds[1]
  linear_order = [first_ind]
  outinds = setdiff(outinds, linear_order)
  while length(outinds) > 1
    sourceinds_list = [Vector{Index}([linear_order..., i]) for i in outinds]
    target_inds, _ = _mincut_inds(tn_pair, out_to_maxweight_ind, sourceinds_list)
    new_ind = setdiff(target_inds, linear_order)[1]
    push!(linear_order, new_ind)
    outinds = setdiff(outinds, [new_ind])
  end
  push!(linear_order, outinds[1])
  return linear_order
end

function _binary_tree_partition_inds_mincut(
  tn_pair::Pair{<:ITensorNetwork,<:ITensorNetwork}, out_to_maxweight_ind::Dict{Index,Index}
)
  outinds = collect(keys(out_to_maxweight_ind))
  @assert length(outinds) >= 1
  if length(outinds) <= 2
    return outinds
  end
  while length(outinds) > 2
    tree_list = collect(powerset(outinds, 2, 2))
    sourceinds_list = [collect(Leaves(i)) for i in tree_list]
    _, i = _mincut_inds(tn_pair, out_to_maxweight_ind, sourceinds_list)
    tree = tree_list[i]
    outinds = setdiff(outinds, tree)
    outinds = vcat([tree], outinds)
  end
  return outinds
end

function _mincut_partitions_costs(
  tn_pair::Pair{<:ITensorNetwork,<:ITensorNetwork},
  out_to_maxweight_ind::Dict{Index,Index},
  sourceinds_list::Vector{<:Vector{<:Index}},
)
  function _mincut_value(tn, sinds, outinds)
    tinds = setdiff(outinds, sinds)
    _, _, cut = _mincut(tn, sinds, tinds)
    return cut
  end
  function _get_weights(source_inds, outinds, maxweight_source_inds, maxweight_outinds)
    mincut_val = _mincut_value(tn_pair.first, source_inds, outinds)
    maxweight_mincut_val = _mincut_value(
      tn_pair.second, maxweight_source_inds, maxweight_outinds
    )
    dist = _distance(tn_pair.first, source_inds)
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
  return weights
end

"""
Find a vector of indices within sourceinds_list yielding the mincut of given tn_pair.
Args:
  tn_pair: a pair of tns (tn1 => tn2), where tn2 is generated via _maxweightoutinds_tn(tn1)
  out_to_maxweight_ind: a dict mapping each out ind in tn1 to out ind in tn2
  sourceinds_list: a list of vector of indices to be considered
Note:
  For each sourceinds in sourceinds_list, we consider its mincut within both tns (tn1, tn2) given in tn_pair.
  The mincut in tn1 represents the rank upper bound when splitting sourceinds with other inds in outinds.
  The mincut in tn2 represents the rank upper bound when the weights of outinds are very large.
  The first mincut upper_bounds the number of non-zero singular values, while the second empirically reveals the
  singular value decay.
  We output the sourceinds where the first mincut value is the minimum, the secound mincut value is also
  the minimum under the condition that the first mincut is optimal, and the sourceinds have the lowest all-pair shortest path.
"""
function _mincut_inds(
  tn_pair::Pair{<:ITensorNetwork,<:ITensorNetwork},
  out_to_maxweight_ind::Dict{Index,Index},
  sourceinds_list::Vector{<:Vector{<:Index}},
)
  weights = _mincut_partitions_costs(tn_pair, out_to_maxweight_ind, sourceinds_list)
  i = argmin(weights)
  return sourceinds_list[i], i
end

function _mps_mincut_partition_cost(tn::ITensorNetwork, inds_vector::Vector{Set})
  @timeit_debug ITensors.timer "_mps_mincut_partition_cost" begin
    inds_vector = map(inds -> collect(inds), inds_vector)
    outinds = vcat(inds_vector...)
    maxweight_tn, out_to_maxweight_ind = _maxweightoutinds_tn(tn, outinds)
    sourceinds_list = [vcat(inds_vector[1:i]...) for i in 1:(length(inds_vector) - 1)]
    weights = _mincut_partitions_costs(
      tn => maxweight_tn, out_to_maxweight_ind, sourceinds_list
    )
    mincut_val = sum([w[1] for w in weights])
    maxweight_mincut_val = sum([w[2] for w in weights])
    dist = sum([w[3] for w in weights])
    return (mincut_val, maxweight_mincut_val, dist)
  end
end
