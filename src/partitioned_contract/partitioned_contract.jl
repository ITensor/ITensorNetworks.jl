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
    tn_leaves = get_leaves(ctree)
    ctrees = topo_sort(ctree; leaves=tn_leaves)
    @info "start _approximate_contract_pre_process"
    ctree_to_igs, ctree_to_adj_tree, ig_to_linear_order = _approximate_contract_pre_process(
      tn_leaves, ctrees
    )
    ctree_to_reference_order = Dict{Vector,Vector}()
    ctree_to_edgeset_order = Dict{Vector,Vector}()
    for leaf in tn_leaves
      reference_order = _mps_partition_inds_set_order(
        ITensorNetwork(leaf), ctree_to_igs[leaf]
      )
      order, _ = mindist_ordering(ctree_to_adj_tree[leaf], reference_order, vectorize(leaf))
      ctree_to_edgeset_order[leaf] = order
    end
    for (ii, c) in enumerate(ctrees)
      @info "$(ii)/$(length(ctrees))", "th pre-process"
      if ctree_to_igs[c] == []
        continue
      end
      ctree_to_reference_order[c], ctree_to_edgeset_order[c] = mindist_ordering(
        ctree_to_adj_tree[c],
        ctree_to_edgeset_order[c[1]],
        ctree_to_edgeset_order[c[2]],
        c[1] in tn_leaves,
        c[2] in tn_leaves,
        vectorize(c),
      )
      # @info "reference ordering", ctree_to_reference_order[c]
      # @info "edge set ordering", ctree_to_edgeset_order[c]
    end
    # return [ITensor(1.0)], 1.0
    # mapping each contraction tree to its contract igs
    ctree_to_contract_igs = Dict{Vector,Vector{IndexGroup}}()
    for c in ctrees
      if c[1] in tn_leaves
        contract_igs = intersect(ctree_to_edgeset_order[c[2]], ctree_to_edgeset_order[c[1]])
        l_igs_c1, r_igs_c1 = split_igs(ctree_to_edgeset_order[c[1]], contract_igs)
        ctree_to_edgeset_order[c[1]] = Vector{IndexGroup}([
          l_igs_c1..., contract_igs..., r_igs_c1...
        ])
      else
        contract_igs = intersect(ctree_to_edgeset_order[c[1]], ctree_to_edgeset_order[c[2]])
        l_igs_c2, r_igs_c2 = split_igs(ctree_to_edgeset_order[c[2]], contract_igs)
        ctree_to_edgeset_order[c[2]] = Vector{IndexGroup}([
          l_igs_c2..., contract_igs..., r_igs_c2...
        ])
      end
      ctree_to_contract_igs[c[1]] = contract_igs
      ctree_to_contract_igs[c[2]] = contract_igs
    end
    # special case when the network contains uncontracted inds
    if haskey(ctree_to_edgeset_order, ctrees[end])
      ctree_to_contract_igs[ctrees[end]] = ctree_to_edgeset_order[ctrees[end]]
    end
    # mapping each contraction tree to a tensor network
    ctree_to_tn_tree = Dict{Vector,Union{Dict{Vector,ITensor},Vector{ITensor}}}()
    # accumulate norm
    log_accumulated_norm = 0.0
    for (ii, c) in enumerate(ctrees)
      t00 = time()
      @info "$(ii)/$(length(ctrees))", "th tree approximation"
      if ctree_to_igs[c] == []
        @assert c == ctrees[end]
        tn1 = get_child_tn(ctree_to_tn_tree, c[1])
        tn2 = get_child_tn(ctree_to_tn_tree, c[2])
        tn = vcat(tn1, tn2)
        out = _optcontract(tn)
        out_nrm = norm(out)
        out /= out_nrm
        return [out], log_accumulated_norm + log(out_nrm)
      end
      tn1 = get_child_tn(ctree_to_tn_tree, c[1])
      tn2 = get_child_tn(ctree_to_tn_tree, c[2])
      # TODO: change new_igs into a vector of igs
      inds_btree = ordered_igs_to_binary_tree(
        ctree_to_edgeset_order[c],
        ctree_to_contract_igs[c],
        ig_to_linear_order;
        ansatz=ansatz,
      )
      ctree_to_tn_tree[c], log_root_norm = approximate_contract_ctree_to_tensor(
        [tn1..., tn2...], inds_btree; cutoff=cutoff, maxdim=maxdim, algorithm=algorithm
      )
      log_accumulated_norm += log_root_norm
      # release the memory
      delete!(ctree_to_tn_tree, c[1])
      delete!(ctree_to_tn_tree, c[2])
      t11 = time() - t00
      @info "time of this contraction is", t11
    end
    tn = vcat(collect(values(ctree_to_tn_tree[ctrees[end]]))...)
    return tn, log_accumulated_norm
  end
end

function _approximate_contract_pre_process(tn_leaves, ctrees)
  @timeit_debug ITensors.timer "_approximate_contract_pre_process" begin
    # mapping each contraction tree to its uncontracted index groups
    ctree_to_igs = Dict{Vector,Vector{IndexGroup}}()
    index_groups = get_index_groups(ctrees[end])
    for c in vcat(tn_leaves, ctrees)
      # TODO: the order here is not optimized
      ctree_to_igs[c] = neighbor_index_groups(c, index_groups)
    end
    ctree_to_path = _get_paths(ctrees[end])
    # mapping each contraction tree to its index adjacency tree
    ctree_to_adj_tree = Dict{Vector,NamedDiGraph{Tuple{Tuple,String}}}()
    for leaf in tn_leaves
      ctree_to_adj_tree[leaf] = _generate_adjacency_tree(
        leaf, ctree_to_path[leaf], ctree_to_igs
      )
    end
    for c in ctrees
      adj_tree = _generate_adjacency_tree(c, ctree_to_path[c], ctree_to_igs)
      if adj_tree != nothing
        ctree_to_adj_tree[c] = adj_tree
        # @info "ctree_to_adj_tree[c]", ctree_to_adj_tree[c]
      end
    end
    # mapping each index group to a linear ordering
    ig_to_linear_order = Dict{IndexGroup,Vector}()
    for leaf in tn_leaves
      for ig in ctree_to_igs[leaf]
        if !haskey(ig_to_linear_order, ig)
          inds_order = _mps_partition_inds_order(ITensorNetwork(leaf), ig.data)
          ig_to_linear_order[ig] = [[i] for i in inds_order]
        end
      end
    end
    return ctree_to_igs, ctree_to_adj_tree, ig_to_linear_order
  end
end