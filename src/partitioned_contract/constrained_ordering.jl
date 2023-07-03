function _constrained_minswap_inds_ordering(
  constraint_tree::NamedDiGraph{Tuple{Tuple,String}},
  ref_ordering::Vector{Set},
  tn::ITensorNetwork,
)
  leaves = leaf_vertices(constraint_tree)
  root = _root(constraint_tree)
  v_to_order = Dict{Tuple{Tuple,String},Vector{IndexGroup}}()
  for v in post_order_dfs_vertices(constraint_tree, root)
    if v in leaves
      v_to_order[v] = [v[1]...]
      continue
    end
    child_orders = Vector{Vector{IndexGroup}}()
    children = child_vertices(constraint_tree, v)
    for inds_tuple in v[1]
      cs = filter(c -> c[1] == inds_tuple, children)
      @assert length(cs) == 1
      push!(child_orders, v_to_order[cs[1]])
    end
    input_order = [n for n in ref_ordering if n in vcat(child_orders...)]
    # Optimize the ordering in child_orders
    if v[2] == "ordered"
      perms = [child_orders, reverse(child_orders)]
      nswaps = [length(_bubble_sort(vcat(p...), input_order)) for p in perms]
      perms = [perms[i] for i in 1:length(perms) if nswaps[i] == min(nswaps...)]
      output_order = _mincut_permutation(perms, tn)
    else
      output_order = _best_perm_greedy(child_orders, input_order, tn)
    end
    v_to_order[v] = vcat(output_order...)
  end
  nswap = length(_bubble_sort(v_to_order[root], ref_ordering))
  return v_to_order[root], nswap
end

function _constrained_minswap_inds_ordering(
  constraint_tree::NamedDiGraph{Tuple{Tuple,String}},
  input_order_1::Vector{Set},
  input_order_2::Vector{Set},
  tn::ITensorNetwork,
)
  # TODO: edge set ordering of tn
end

function _mincut_permutation(perms::Vector{<:Vector}, tn::ITensorNetwork)
  if length(perms) == 1
    return perms[1]
  end
  mincuts_dist = map(p -> _mps_mincut_partition_cost(tn, p), perms)
  return perms[argmin(mincuts_dist)]
end

function _best_perm_greedy(vs::Vector{<:Vector}, order::Vector, tn::ITensorNetwork)
  ordered_vs = [vs[1]]
  for v in vs[2:end]
    perms = [insert!(copy(ordered_vs), i, v) for i in 1:(length(ordered_vs) + 1)]
    suborder = filter(n -> n in vcat(perms[1]...), order)
    nswaps = map(p -> length(_bubble_sort(vcat(p...), suborder)), perms)
    perms = [perms[i] for i in 1:length(perms) if nswaps[i] == min(nswaps...)]
    ordered_vs = _mincut_permutation(perms, tn)
  end
  return ordered_vs
end
