function partitioned_contract(
  partition::DataGraph,
  contraction_tree::NamedDiGraph;
  ansatz="mps",
  approx_itensornetwork_alg="density_matrix",
  cutoff=1e-15,
  maxdim=10000,
  swap_size=1,
  contraction_sequence_alg,
  contraction_sequence_kwargs,
)
  @timeit_debug ITensors.timer "partitioned_contract" begin
    leaves = leaf_vertices(contraction_tree)
    traversal = post_order_dfs_vertices(contraction_tree, _root(contraction_tree))
    contractions = setdiff(traversal, leaves)
    p_edge_to_ordered_inds = _ind_orderings(partition)
    # Build the orderings used for the ansatz tree.
    # For each tuple in `v_to_ordered_p_edges`, the first item is the
    # reference ordering of uncontracted edges for the contraction `v`,
    # the second item is the target ordering, and the third item is the
    # ordering of part of the uncontracted edges that are to be contracted
    # in the next contraction (first contraction in the path of `v`).
    v_to_ordered_p_edges = Dict{Tuple,Tuple}()
    for (ii, v) in enumerate(traversal)
      @info "$(ii)/$(length(traversal)) th ansatz tree construction"
      p_leaves = vcat(v[1:(end - 1)]...)
      tn = ITensorNetwork(mapreduce(l -> Vector{ITensor}(partition[l]), vcat, p_leaves))
      path = filter(u -> issubset(p_leaves, u[1]) || issubset(p_leaves, u[2]), contractions)
      p_edges = _neighbor_edges(partition, p_leaves)
      # Get the reference ordering and target ordering of `v`
      v_inds = map(e -> Set(p_edge_to_ordered_inds[e]), p_edges)
      constraint_tree = _adjacency_tree(v, path, partition, p_edge_to_ordered_inds)
      if v in leaves
        ref_inds_ordering = _mps_partition_inds_order(tn, v_inds; alg="top_down")
        inds_ordering = _constrained_minswap_inds_ordering(
          constraint_tree, ref_inds_ordering, tn
        )
      else
        c1, c2 = child_vertices(contraction_tree, v)
        c1_inds_ordering = map(
          e -> Set(p_edge_to_ordered_inds[e]), v_to_ordered_p_edges[c1][2]
        )
        c2_inds_ordering = map(
          e -> Set(p_edge_to_ordered_inds[e]), v_to_ordered_p_edges[c2][2]
        )
        ref_inds_ordering, inds_ordering = _constrained_minswap_inds_ordering(
          constraint_tree, c1_inds_ordering, c2_inds_ordering, tn
        )
      end
      ref_p_edges = p_edges[_findperm(v_inds, ref_inds_ordering)]
      p_edges = p_edges[_findperm(v_inds, inds_ordering)]
      # Update the contracted ordering and `v_to_ordered_p_edges[v]`.
      # `sibling` is the vertex to be contracted with `v` next.
      # Note: the contracted ordering in `ref_p_edges` is not changed,
      # since `ref_p_edges` focuses on minimizing the number of swaps
      # rather than making later contractions efficient.
      sibling = setdiff(child_vertices(contractions, path[1]), [v])[1]
      if haskey(v_to_ordered_p_edges, sibling)
        contract_edges = v_to_ordered_p_edges[sibling][3]
        @assert(_is_neighbored_subset(p_edges, Set(contract_edges)))
        p_edges = _replace_subarray(p_edges, contract_edges)
      else
        p_leaves_2 = vcat(sibling[1:(end - 1)]...)
        p_edges_2 = _neighbor_edges(partition, p_leaves_2)
        contract_edges = intersect(p_edges, p_edges_2)
      end
      v_to_ordered_p_edges[v] = (ref_p_edges, p_edges, contract_edges)
    end
    # start approximate contraction
    v_to_tn = Dict{Tuple,ITensorNetwork}()
    for v in leaves
      @assert length(v[1]) == 1
      v_to_tn[v] = partition[v[1][1]]
    end
    log_acc_norm = 0.0
    for (ii, v) in enumerate(contractions)
      @info "$(ii)/$(length(contractions)) th tree approximation"
      c1, c2 = child_vertices(contraction_tree, v)
      tn = ITensorNetwork()
      ts = vcat(Vector{ITensor}(v_to_tn[c1]), Vector{ITensor}(v_to_tn[c2]))
      for (i, t) in enumerate(ts)
        add_vertex!(tn, i)
        tn[i] = t
      end
      ref_p_edges, p_edges, contract_edges = v_to_ordered_p_edges[v]
      if p_edges == []
        @assert v == contractions[end]
        out = _optcontract(
          Vector{ITensor}(tn);
          contraction_sequence_alg=contraction_sequence_alg,
          contraction_sequence_kwargs=contraction_sequence_kwargs,
        )
        out_nrm = norm(out)
        out /= out_nrm
        return ITensorNetwork(out), log_acc_norm + log(out_nrm)
      end
      for edge_order in _intervals(ref_p_edges, p_edges; size=swap_size)
        inds_ordering = map(e -> p_edge_to_ordered_inds[e], edge_order)
        # `ortho_center` denotes the number of edges arranged at
        # the left of the center vertex.
        ortho_center = if edge_order == p_edges
          _ortho_center(p_edges, contract_edges)
        else
          div(length(edge_order), 2, RoundDown)
        end
        tn, log_norm = approx_itensornetwork(
          tn,
          _ansatz_tree(inds_ordering, ansatz, ortho_center);
          alg=approx_itensornetwork_alg,
          cutoff=cutoff,
          maxdim=maxdim,
          contraction_sequence_alg=contraction_sequence_alg,
          contraction_sequence_kwargs=contraction_sequence_kwargs,
        )
        log_acc_norm += log_norm
      end
      v_to_tn[v] = tn
    end
    return v_to_tn[contractions[end]], log_acc_norm
  end
end

function _ind_orderings(partition::DataGraph)
  # input_tn = # TODO
  # TODO: use `_mps_partition_inds_order` with default `alg` and `backend`
end

function _ansatz_tree(inds_orderings::Vector, ansatz::String, ortho_center::Integer)
  @assert ansatz in ["comb", "mps"]
  nested_vec_left, nested_vec_right = _nested_vector_pair(
    Algorithm(ansatz), inds_orderings, ortho_center
  )
  if nested_vec_left == [] || nested_vec_right == []
    nested_vec = nested_vec_left == [] ? nested_vec_right : nested_vec_left
  else
    nested_vec = [nested_vec_left, nested_vec_right]
  end
  return _nested_vector_to_digraph(nested_vec)
end

function _nested_vector_pair(
  ::Algorithm"mps", inds_orderings::Vector, ortho_center::Integer
)
  nested_vec_left = line_to_tree(vcat(inds_orderings[1:ortho_center]...))
  nested_vec_right = line_to_tree(reverse(vcat(inds_orderings[(ortho_center + 1):end]...)))
  return nested_vec_left, nested_vec_right
end

function _nested_vector_pair(
  ::Algorithm"comb", inds_orderings::Vector, ortho_center::Integer
)
  nested_vec_left = line_to_tree(
    map(ig -> line_to_tree(ig), inds_orderings[1:ortho_center])
  )
  nested_vec_right = line_to_tree(
    map(ig -> line_to_tree(ig), inds_orderings[end:-1:(ortho_center + 1)])
  )
  return nested_vec_left, nested_vec_right
end

function _ortho_center(edges::Vector, contract_edges::Vector)
  left_edges, right_edges = _split_array(edges, contract_edges)
  if length(left_edges) < length(right_edges)
    return length(left_edges) + length(contract_edges)
  end
  return length(left_edges)
end

function _permute(v::Vector, perms)
  v = copy(v)
  for p in perms
    temp = v[p]
    v[p] = v[p + 1]
    v[p + 1] = temp
  end
  return v
end

function _intervals(v1::Vector, v2::Vector; size)
  if v1 == v2
    return [v2]
  end
  perms_list = collect(Iterators.partition(_bubble_sort(v1, v2), size))
  out = [v1]
  current = v1
  for perms in perms_list
    v = _permute(current, perms)
    push!(out, v)
    current = v
  end
  return out
end
