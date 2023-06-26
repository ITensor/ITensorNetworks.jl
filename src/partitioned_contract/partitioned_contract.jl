function partitioned_contract(
  partition::DataGraph,
  contraction_tree::NamedDiGraph;
  ansatz="mps",
  approx_itensornetwork_alg="density_matrix",
  cutoff=1e-15,
  maxdim=10000,
  contraction_sequence_alg,
  contraction_sequence_kwargs,
)
  @timeit_debug ITensors.timer "partitioned_contract" begin
    leaves = leaf_vertices(contraction_tree)
    traversal = post_order_dfs_vertices(contraction_tree, _root(contraction_tree))
    contractions = setdiff(traversal, leaves)
    p_edge_to_ordered_inds = _ind_orderings(partition)
    # build the orderings used for the ansatz tree
    # For each pair in `v_to_ordered_p_edges`, the first item
    # is the ordering of uncontracted edges for the contraction `v`,
    # and the second item is the ordering of part of the uncontracted edges
    # that are to be contracted in the next contraction (first contraction
    # in the path of `v`).
    v_to_ordered_p_edges = Dict{Tuple,Pair}()
    for (ii, v) in enumerate(contractions)
      @info "$(ii)/$(length(contractions)) th ansatz tree construction"
      p_leaves = [v[1]..., v[2]...]
      tn = ITensorNetwork(mapreduce(l -> Vector{ITensor}(partition[l]), vcat, p_leaves))
      path = filter(u -> issubset(p_leaves, u[1]) || issubset(p_leaves, u[2]), contractions)
      p_edges = _neighbor_edges(partition, p_leaves)
      inds_set = [Set(p_edge_to_ordered_inds[e]) for e in p_edges]
      ordering = _constrained_minswap_inds_ordering(inds_set, tn, path)
      p_edges = p_edges[sortperm(ordering)]
      # update the contracted ordering and `v_to_ordered_p_edges[v]`.
      # path[1] is the vertex to be contracted with `v` next.
      if haskey(v_to_ordered_p_edges, path[1])
        contract_edges = v_to_ordered_p_edges[path[1]].second
        @assert(_is_neighbored_subset(p_edges, Set(contract_edges)))
        p_edges = _replace_subarray(p_edges, contract_edges)
      else
        p_leaves_2 = [path[1][1]..., path[1][2]...]
        p_edges_2 = _neighbor_edges(partition, p_leaves_2)
        contract_edges = intersect(p_edges, p_edges_2)
        contract_edges = filter(e -> e in contract_edges, p_edges)
      end
      v_to_ordered_p_edges[v] = Pair(p_edges, contract_edges)
    end
    # start approx_itensornetwork
    v_to_tn = Dict{Tuple,ITensorNetwork}()
    for v in leaves
      @assert length(v[1]) == 1
      v_to_tn[v] = partition[v[1][1]]
    end
    log_accumulated_norm = 0.0
    for (ii, v) in enumerate(contractions)
      @info "$(ii)/$(length(contractions)) th tree approximation"
      c1, c2 = child_vertices(contraction_tree, v)
      tn = disjoint_union(v_to_tn[c1], v_to_tn[c2])
      # TODO: rename tn since the names will be too long.
      p_edges = v_to_ordered_p_edges[v].first
      if p_edges == []
        # TODO: edge case with output being a scalar
      end
      inds_orderings = [p_edge_to_ordered_inds[e] for e in p_edges]
      v_to_tn[v], log_root_norm = approx_itensornetwork(
        tn,
        _ansatz_tree(inds_orderings, _ortho_center(v_to_ordered_p_edges[v]), ansatz);
        alg=approx_itensornetwork_alg,
        cutoff=cutoff,
        maxdim=maxdim,
        contraction_sequence_alg=contraction_sequence_alg,
        contraction_sequence_kwargs=contraction_sequence_kwargs,
      )
      log_accumulated_norm += log_root_norm
    end
  end
end

function _is_neighbored_subset(v::Vector, set::Set)
  if set == Set()
    return true
  end
  @assert issubset(set, Set(v))
  i_begin = 1
  while !(i_begin in set)
    i_begin += 1
  end
  i_end = length(v)
  while !(i_end in set)
    i_end -= 1
  end
  return Set(v[i_begin:i_end]) == set
end

# replace the subarray of `v1` with `v2`
function _replace_subarray(v1::Vector, v2::Vector)
  if v2 == []
    return v1
  end
  v1 = copy(v1)
  num = 0
  for i in 1:length(v1)
    if v1[i] in v2
      num += 1
      v1[i] = v2[num]
    end
  end
  @assert num == length(v2)
  return v1
end

function _neighbor_edges(graph, vs)
  return filter(
    e -> (e.src in vs && !(e.dst in vs)) || (e.dst in vs && !(e.src in vs)), edges(graph)
  )
end

function _ind_orderings(partition::DataGraph)
  input_tn = # TODO
end

function _constrained_mincost_inds_ordering(inds_set::Set, tn::ITensorNetwork, path::Vector)
  # TODO: edge set ordering of tn
end

function _ansatz_tree(inds_orderings::Vector, ortho_center::Integer, ansatz::String)
  # TODO
end

function _ortho_center(edges_to_contract_edges)
  # TODO
end

# function ordered_igs_to_binary_tree(ordered_igs, contract_igs, ig_to_linear_order; ansatz)
#   @assert ansatz in ["comb", "mps"]
#   @timeit_debug ITensors.timer "ordered_igs_to_binary_tree" begin
#     if contract_igs == []
#       @info "contract_igs is empty vector"
#     end
#     # @assert contract_igs != []
#     left_igs, right_igs = split_igs(ordered_igs, contract_igs)
#     if ansatz == "comb"
#       return ordered_igs_to_binary_tree_comb(
#         left_igs, right_igs, contract_igs, ig_to_linear_order
#       )
#     elseif ansatz == "mps"
#       return ordered_igs_to_binary_tree_mps(
#         left_igs, right_igs, contract_igs, ig_to_linear_order
#       )
#     end
#   end
# end

# function ordered_igs_to_binary_tree(igs, ig_to_linear_order; ansatz, direction)
#   @assert ansatz in ["comb", "mps"]
#   @assert direction in ["left", "right"]
#   if ansatz == "comb"
#     return line_to_tree([line_to_tree(ig_to_linear_order[ig]) for ig in igs])
#   end
#   if direction == "left"
#     order = vcat([ig_to_linear_order[ig] for ig in igs]...)
#     return line_to_tree(order)
#   else
#     # First reverse get the order from middle to boundary,
#     # and second reverse get the overall inds order from boundary to middle.
#     order = vcat([ig_to_linear_order[ig] for ig in reverse(igs)]...)
#     return line_to_tree(reverse(order))
#   end
# end

# function ordered_igs_to_binary_tree_mps(
#   left_igs, right_igs, contract_igs, ig_to_linear_order
# )
#   left_order = get_leaves([ig_to_linear_order[ig] for ig in left_igs])
#   right_order = get_leaves([ig_to_linear_order[ig] for ig in right_igs])
#   contract_order = get_leaves([ig_to_linear_order[ig] for ig in contract_igs])
#   if length(left_order) <= length(right_order)
#     left_order = [left_order..., contract_order...]
#   else
#     right_order = [contract_order..., right_order...]
#   end
#   return merge_tree(line_to_tree(left_order), line_to_tree(reverse(right_order)))
# end

# function ordered_igs_to_binary_tree_comb(
#   left_igs, right_igs, contract_igs, ig_to_linear_order
# )
#   tree_1 = ordered_igs_to_binary_tree(
#     left_igs, ig_to_linear_order; ansatz="comb", direction="left"
#   )
#   tree_contract = ordered_igs_to_binary_tree(
#     contract_igs, ig_to_linear_order; ansatz="comb", direction="left"
#   )
#   tree_2 = ordered_igs_to_binary_tree(
#     reverse(right_igs), ig_to_linear_order; ansatz="comb", direction="left"
#   )
#   # make the binary tree more balanced to save tree approximation cost
#   if tree_1 == []
#     return merge_tree(merge_tree(tree_1, tree_contract), tree_2)
#   end
#   if tree_2 == []
#     return merge_tree(tree_1, merge_tree(tree_contract, tree_2))
#   end
#   if length(vectorize(tree_1)) <= length(vectorize(tree_2))
#     return merge_tree(merge_tree(tree_1, tree_contract), tree_2)
#   else
#     return merge_tree(tree_1, merge_tree(tree_contract, tree_2))
#   end
# end
