function boundary_state(v::Tuple{Tuple,String}, adj_igs::Set)
  if Set(Leaves(v[1])) == adj_igs
    return "all"
  end
  if v[2] == "unordered"
    filter_children = filter(c -> issubset(adj_igs, Set(Leaves(c))), v[1])
    # length(filter_children) < 1 means adj_igs is distributed in multiple children
    @assert length(filter_children) <= 1
    if length(filter_children) == 1
      return "middle"
    end
    # TODO: if more than 1 children contain adj_igs, currently we don't reorder the
    # leaves. This may need to be optimized later.
    return "invalid"
  end
  @assert length(v[1]) >= 2
  for i in 1:(length(v[1]) - 1)
    leaves = vcat([Set(Leaves(c)) for c in v[1][1:i]]...)
    if Set(leaves) == adj_igs
      return "left"
    end
  end
  for i in 2:length(v[1])
    leaves = vcat([Set(Leaves(c)) for c in v[1][i:end]]...)
    if Set(leaves) == adj_igs
      return "right"
    end
  end
  return "invalid"
end

function reorder_to_boundary!(
  adj_tree::NamedDiGraph{Tuple{Tuple,String}},
  v::Tuple{Tuple,String},
  target_child::Tuple{Tuple,String};
  direction="right",
)
  new_v = v
  children = child_vertices(adj_tree, v)
  remain_children = setdiff(children, [target_child])
  @assert length(remain_children) >= 1
  if length(remain_children) == 1
    remain_child = remain_children[1]
    if direction == "right"
      new_v = ((remain_child[1], target_child[1]), "ordered")
    else
      new_v = ((target_child[1], remain_child[1]), "ordered")
    end
    if new_v != v
      _add_vertex_edges!(
        adj_tree, new_v; children=children, parent=parent_vertex(adj_tree, v)
      )
      rem_vertex!(adj_tree, v)
    end
  else
    new_child = (Tuple([v[1] for v in remain_children]), "unordered")
    _add_vertex_edges!(adj_tree, new_child; children=remain_children, parent=v)
    if direction == "right"
      new_v = ((new_child[1], target_child[1]), "ordered")
    else
      new_v = ((target_child[1], new_child[1]), "ordered")
    end
    _add_vertex_edges!(
      adj_tree, new_v; children=[new_child, target_child], parent=parent_vertex(adj_tree, v)
    )
    rem_vertex!(adj_tree, v)
  end
  return new_v
end

function _add_vertex_edges!(
  adj_tree::NamedDiGraph{Tuple{Tuple,String}}, v; children=[], parent=nothing
)
  add_vertex!(adj_tree, v)
  if parent != nothing
    add_edge!(adj_tree, parent => v)
  end
  for c in children
    add_edge!(adj_tree, v => c)
  end
end

"""
reorder adj_tree based on adj_igs
"""
function reorder!(
  adj_tree::NamedDiGraph{Tuple{Tuple,String}},
  root::Tuple{Tuple,String},
  adj_igs::Set;
  boundary="right",
)
  @assert boundary in ["left", "right"]
  if boundary_state(root, adj_igs) == "all"
    return false, root
  end
  sub_tree = subgraph(v -> issubset(Set(Leaves(v[1])), Set(Leaves(root[1]))), adj_tree)
  traversal = post_order_dfs_vertices(sub_tree, root)
  path = [v for v in traversal if issubset(adj_igs, Set(Leaves(v[1])))]
  new_root = root
  # get the boundary state
  v_to_state = Dict{Tuple{Tuple,String},String}()
  for v in path
    state = boundary_state(v, adj_igs)
    if state == "invalid"
      return false, root
    end
    v_to_state[v] = state
  end
  for v in path
    children = child_vertices(adj_tree, v)
    # reorder
    if v_to_state[v] in ["left", "right"] && v_to_state[v] != boundary
      @assert v[2] == "ordered"
      new_v = (reverse(v[1]), v[2])
      new_root = (v == root) ? new_v : new_root
      _add_vertex_edges!(
        adj_tree, new_v; children=children, parent=parent_vertex(adj_tree, v)
      )
      rem_vertex!(adj_tree, v)
    elseif v_to_state[v] == "middle"
      @assert v[2] == "unordered"
      target_child = filter(c -> issubset(adj_igs, Set(Leaves(c[1]))), children)
      @assert length(target_child) == 1
      new_v = reorder_to_boundary!(adj_tree, v, target_child[1]; direction=boundary)
      new_root = (v == root) ? new_v : new_root
    end
  end
  return true, new_root
end

# Update both keys and values in igs_to_adjacency_tree based on adjacent_igs
# NOTE: keys of `igs_to_adjacency_tree` are target igs, not those adjacent to ancestors
function update_adjacency_tree!(
  adjacency_tree::NamedDiGraph{Tuple{Tuple,String}}, adjacent_igs::Set
)
  @timeit_debug ITensors.timer "update_adjacency_tree" begin
    root_v_to_adjacent_igs = Dict{Tuple{Tuple,String},Set}()
    for r in _roots(adjacency_tree)
      root_igs = Set(Leaves(r[1]))
      common_igs = intersect(adjacent_igs, root_igs)
      if common_igs != Set()
        root_v_to_adjacent_igs[r] = common_igs
      end
    end
    if length(root_v_to_adjacent_igs) == 1
      return nothing
    end
    # if at least 3: for now just put everything together
    if length(root_v_to_adjacent_igs) >= 3
      __roots = keys(root_v_to_adjacent_igs)
      new_v = (Tuple([r[1] for r in __roots]), "unordered")
      _add_vertex_edges!(adjacency_tree, new_v; children=__roots)
      return nothing
    end
    # if 2: assign adjacent_igs to boundary of root_igs (if possible), then concatenate
    v1, v2 = collect(keys(root_v_to_adjacent_igs))
    reordered_1, update_v1 = reorder!(
      adjacency_tree, v1, root_v_to_adjacent_igs[v1]; boundary="right"
    )
    reordered_2, update_v2 = reorder!(
      adjacency_tree, v2, root_v_to_adjacent_igs[v2]; boundary="left"
    )
    cs1 = child_vertices(adjacency_tree, update_v1)
    cs2 = child_vertices(adjacency_tree, update_v2)
    if (!reordered_1) && (!reordered_2)
      new_v = ((update_v1[1], update_v2[1]), "unordered")
      _add_vertex_edges!(adjacency_tree, new_v; children=[update_v1, update_v2])
    elseif (reordered_2)
      new_v = ((update_v1[1], update_v2[1]...), "ordered")
      _add_vertex_edges!(adjacency_tree, new_v; children=[update_v1, cs2...])
      rem_vertex!(adjacency_tree, update_v2)
    elseif (reordered_1)
      new_v = ((update_v1[1]..., update_v2[1]), "ordered")
      _add_vertex_edges!(adjacency_tree, new_v; children=[update_v2, cs1...])
      rem_vertex!(adjacency_tree, update_v1)
    else
      new_v = ((update_v1[1]..., update_v2[1]...), "ordered")
      _add_vertex_edges!(adjacency_tree, new_v; children=[cs1..., cs2...])
      rem_vertex!(adjacency_tree, update_v1)
      rem_vertex!(adjacency_tree, update_v2)
    end
  end
end

# Generate the adjacency tree of a contraction tree
# TODO: add test
function _adjacency_tree(v::Tuple, path::Vector, partition::DataGraph, p_edge_to_inds::Dict)
  @timeit_debug ITensors.timer "_generate_adjacency_tree" begin
    # mapping each index group to adjacent input igs
    ig_to_input_adj_igs = Dict{Any,Set}()
    # mapping each igs to an adjacency tree
    adjacency_tree = NamedDiGraph{Tuple{Tuple,String}}()
    p_leaves = vcat(v[1:(end - 1)]...)
    p_edges = _neighbor_edges(partition, p_leaves)
    for ig in map(e -> Set(p_edge_to_inds[e]), p_edges)
      ig_to_input_adj_igs[ig] = Set([ig])
      v = ((ig,), "unordered")
      add_vertex!(adjacency_tree, v)
    end
    for contraction in path
      children = child_vertices(contractions, path[1])
      ancester = filter(u -> p_leaves in vcat(u[1:(end - 1)]...), children)[1]
      sibling = setdiff(children, [ancester])[1]
      ancester_igs = map(
        e -> Set(p_edge_to_inds[e]),
        _neighbor_edges(partition, vcat(ancester[1:(end - 1)]...)),
      )
      sibling_igs = map(
        e -> Set(p_edge_to_inds[e]),
        _neighbor_edges(partition, vcat(sibling[1:(end - 1)]...)),
      )
      inter_igs = intersect(ancester_igs, sibling_igs)
      new_igs = setdiff(sibling_igs, inter_igs)
      adjacent_igs = union([ig_to_input_adj_igs[ig] for ig in inter_igs]...)
      # `inter_igs != []` means it's a tensor product
      if inter_igs != []
        update_adjacency_tree!(adjacency_tree, adjacent_igs)
      end
      for ig in new_igs
        ig_to_input_adj_igs[ig] = adjacent_igs
      end
      # @info "adjacency_tree", adjacency_tree
      if length(_roots(adjacency_tree)) == 1
        return adjacency_tree
      end
    end
    __roots = _roots(adjacency_tree)
    if length(__roots) > 1
      new_v = (Tuple([r[1] for r in __roots]), "unordered")
      _add_vertex_edges!(adjacency_tree, new_v; children=__roots)
    end
    return adjacency_tree
  end
end

function _constrained_mincost_inds_ordering(
  inds_set::Vector{Set},
  constraint_tree::NamedDiGraph{Tuple{Tuple,String}},
  tn::ITensorNetwork,
)
  # TODO: edge set ordering of tn
end

function _constrained_mincost_inds_ordering(
  inds_set::Vector{Set},
  constraint_tree::NamedDiGraph{Tuple{Tuple,String}},
  tn::ITensorNetwork,
  input_order_1::Vector{Set},
  input_order_2::Vector{Set},
)
  # TODO: edge set ordering of tn
end
