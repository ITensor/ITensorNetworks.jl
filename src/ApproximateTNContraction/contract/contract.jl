using Graphs, GraphsFlows, Combinatorics, SimpleWeightedGraphs
using GraphRecipes, Plots
using OMEinsumContractionOrders
using ITensorNetworks: contraction_sequence

function Base.show(io::IO, tensor::ITensor)
  return print(io, string(inds(tensor)))
end

include("tree_utils.jl")
include("index_group.jl")
include("tensornetwork_graph.jl")
include("mincut_tree.jl")
include("tree_embedding.jl")

function optcontract(t_list::Vector)
  @timeit timer "optcontract" begin
    # TODO: make this support orthotensor
    if length(t_list) == 0
      return ITensor(1.0)
    end
    @assert t_list isa Vector{OrthogonalITensor}
    t_list = get_tensors(t_list)
    # for t in t_list
    #   @info "size of t is", size(t)
    # end
    @timeit timer "contraction_sequence" begin
      seq = contraction_sequence(t_list; alg="greedy")
    end
    @timeit timer "contract" begin
      output = contract(t_list; sequence=seq)
    end
    return OrthogonalITensor(output)
  end
end

approximate_contract(tn::ITensor, inds_groups; kwargs...) = [tn], 0.0

approximate_contract(tn::OrthogonalITensor, inds_groups; kwargs...) = [tn], 0.0

function approximate_contract(tn::Vector{ITensor}, inds_btree=nothing; kwargs...)
  out, log_norm = approximate_contract(orthogonal_tensors(tn), inds_btree; kwargs...)
  return get_tensors(out), log_norm
end

function approximate_contract(tn::Vector{OrthogonalITensor}, inds_btree=nothing; kwargs...)
  ctree_to_tensor, log_root_norm = approximate_contract_ctree_to_tensor(
    tn, inds_btree; kwargs...
  )
  return Vector{OrthogonalITensor}(vcat(collect(values(ctree_to_tensor))...)), log_root_norm
end

function approximate_contract_ctree_to_tensor(
  tn::Vector{OrthogonalITensor},
  inds_btree=nothing;
  cutoff,
  maxdim,
  maxsize=10^15,
  algorithm="mincut-mps",
)
  uncontract_inds = noncommoninds(tn...)
  allinds = collect(Set(mapreduce(t -> collect(inds(t)), vcat, tn)))
  innerinds = setdiff(allinds, uncontract_inds)
  if length(uncontract_inds) <= 2
    if inds_btree == nothing
      inds_btree = [[i] for i in uncontract_inds]
    end
    return Dict{Vector,OrthogonalITensor}(inds_btree => optcontract(tn)), 0.0
  end
  # # cases where tn is a tree, or contains 2 disconnected trees
  # if length(innerinds) <= length(tn) - 1
  #   # TODO
  #   return tn
  # end
  # # TODO: may want to remove this
  # if inds_groups != nothing
  #   deltainds = vcat(filter(g -> length(g) > 1, inds_groups)...)
  #   deltas, tnprime, _ = split_deltas(deltainds, tn)
  #   tn = Vector{ITensor}(vcat(deltas, tnprime))
  # end
  if inds_btree == nothing
    inds_btree = inds_binary_tree(get_tensors(tn), nothing; algorithm=algorithm)
  end
  # tree_approximation(tn, inds_btree; cutoff=cutoff, maxdim=maxdim)
  embedding = tree_embedding(tn, inds_btree)
  tn = Vector{OrthogonalITensor}(vcat(collect(values(embedding))...))
  i2 = noncommoninds(tn...)
  @assert (length(uncontract_inds) == length(i2))
  @timeit timer "tree_approximation_cache" begin
    return tree_approximation_cache(
      embedding, inds_btree; cutoff=cutoff, maxdim=maxdim, maxsize=maxsize
    )
  end
end

function uncontractinds(tn)
  if tn isa ITensor
    return inds(tn)
  else
    return noncommoninds(vectorize(tn)...)
  end
end

# Note that the children ordering matters here.
mutable struct IndexAdjacencyTree
  children::Union{Vector{IndexAdjacencyTree},Vector{IndexGroup}}
  fixed_direction::Bool
  fixed_order::Bool
end

function Base.copy(tree::IndexAdjacencyTree)
  node_to_copynode = Dict{IndexAdjacencyTree,IndexAdjacencyTree}()
  for node in topo_sort(tree; type=IndexAdjacencyTree)
    if node.children isa Vector{IndexGroup}
      node_to_copynode[node] = IndexAdjacencyTree(
        node.children, node.fixed_direction, node.fixed_order
      )
      continue
    end
    copynode = IndexAdjacencyTree(
      [node_to_copynode[n] for n in node.children], node.fixed_direction, node.fixed_order
    )
    node_to_copynode[node] = copynode
  end
  return node_to_copynode[tree]
end

function Base.show(io::IO, tree::IndexAdjacencyTree)
  out_str = "\n"
  stack = [tree]
  node_to_level = Dict{IndexAdjacencyTree,Int}()
  node_to_level[tree] = 0
  # pre-order traversal
  while length(stack) != 0
    node = pop!(stack)
    indent_vec = ["  " for _ in 1:node_to_level[node]]
    indent = string(indent_vec...)
    if node.children isa Vector{IndexGroup}
      for c in node.children
        out_str = out_str * indent * string(c) * "\n"
      end
    else
      out_str =
        out_str *
        indent *
        "AdjTree: [fixed_direction]: " *
        string(node.fixed_direction) *
        " [fixed_order]: " *
        string(node.fixed_order) *
        "\n"
      for c in node.children
        node_to_level[c] = node_to_level[node] + 1
        push!(stack, c)
      end
    end
  end
  return print(io, out_str)
end

function IndexAdjacencyTree(index_group::IndexGroup)
  return IndexAdjacencyTree([index_group], false, false)
end

function get_adj_tree_leaves(tree::IndexAdjacencyTree)
  if tree.children isa Vector{IndexGroup}
    return tree.children
  end
  leaves = [get_adj_tree_leaves(c) for c in tree.children]
  return vcat(leaves...)
end

function Base.contains(adj_tree::IndexAdjacencyTree, adj_igs::Set{IndexGroup})
  leaves = Set(get_adj_tree_leaves(adj_tree))
  return issubset(adj_igs, leaves)
end

function Base.iterate(x::IndexAdjacencyTree)
  return iterate(x, 1)
end

function Base.iterate(x::IndexAdjacencyTree, index)
  if index > length(x.children)
    return nothing
  end
  return x.children[index], index + 1
end

function boundary_state(ancestor::IndexAdjacencyTree, adj_igs::Set{IndexGroup})
  if ancestor.children isa Vector{IndexGroup}
    return "all"
  end
  if !ancestor.fixed_order
    filter_children = filter(a -> contains(a, adj_igs), ancestor.children)
    @assert length(filter_children) <= 1
    if length(filter_children) == 1
      return "middle"
    elseif Set(get_adj_tree_leaves(ancestor)) == adj_igs
      return "all"
    else
      return "invalid"
    end
  end
  @assert length(ancestor.children) >= 2
  if contains(ancestor.children[1], adj_igs)
    return "left"
  elseif contains(ancestor.children[end], adj_igs)
    return "right"
  elseif Set(get_adj_tree_leaves(ancestor)) == adj_igs
    return "all"
  else
    return "invalid"
  end
end

function reorder_to_right!(
  ancestor::IndexAdjacencyTree, filter_children::Vector{IndexAdjacencyTree}
)
  remain_children = setdiff(ancestor.children, filter_children)
  @assert length(filter_children) >= 1
  @assert length(remain_children) >= 1
  if length(remain_children) == 1
    new_child1 = remain_children[1]
  else
    new_child1 = IndexAdjacencyTree(remain_children, false, false)
  end
  if length(filter_children) == 1
    new_child2 = filter_children[1]
  else
    new_child2 = IndexAdjacencyTree(filter_children, false, false)
  end
  ancestor.children = [new_child1, new_child2]
  return ancestor.fixed_order = true
end

"""
reorder adj_tree based on adj_igs
"""
function reorder!(adj_tree::IndexAdjacencyTree, adj_igs::Set{IndexGroup}; boundary="right")
  @assert boundary in ["left", "right"]
  if boundary_state(adj_tree, adj_igs) == "all"
    return false
  end
  adj_trees = topo_sort(adj_tree; type=IndexAdjacencyTree)
  ancestors = [tree for tree in adj_trees if contains(tree, adj_igs)]
  ancestor_to_state = Dict{IndexAdjacencyTree,String}()
  # get the boundary state
  for ancestor in ancestors
    state = boundary_state(ancestor, adj_igs)
    if state == "invalid"
      return false
    end
    ancestor_to_state[ancestor] = state
  end
  # update ancestors
  for ancestor in ancestors
    # reorder
    if ancestor_to_state[ancestor] == "left"
      ancestor.children = reverse(ancestor.children)
    elseif ancestor_to_state[ancestor] == "middle"
      @assert ancestor.fixed_order == false
      filter_children = filter(a -> contains(a, adj_igs), ancestor.children)
      reorder_to_right!(ancestor, filter_children)
    end
    # merge
    if ancestor.fixed_order && ancestor.children isa Vector{IndexAdjacencyTree}
      new_children = Vector{IndexAdjacencyTree}()
      for child in ancestor.children
        if !child.fixed_order
          push!(new_children, child)
        else
          push!(new_children, child.children...)
        end
      end
      ancestor.children = new_children
    end
  end
  # check boundary
  if boundary == "left"
    for ancestor in ancestors
      ancestor.children = reverse(ancestor.children)
    end
  end
  return true
end

# Update both keys and values in igs_to_adjacency_tree based on list_adjacent_igs
function update_igs_to_adjacency_tree!(
  list_adjacent_igs::Vector, igs_to_adjacency_tree::Dict{Set{IndexGroup},IndexAdjacencyTree}
)
  function update!(root_igs, adjacent_igs)
    if !haskey(root_igs_to_adjacent_igs, root_igs)
      root_igs_to_adjacent_igs[root_igs] = adjacent_igs
    else
      val = root_igs_to_adjacent_igs[root_igs]
      root_igs_to_adjacent_igs[root_igs] = union(val, adjacent_igs)
    end
  end
  @timeit timer "update_igs_to_adjacency_tree" begin
    # get each root igs, get the adjacent igs needed. TODO: do we need to consider boundaries here?
    root_igs_to_adjacent_igs = Dict{Set{IndexGroup},Set{IndexGroup}}()
    for adjacent_igs in list_adjacent_igs
      for root_igs in keys(igs_to_adjacency_tree)
        if issubset(adjacent_igs, root_igs)
          update!(root_igs, adjacent_igs)
        end
      end
    end
    if length(root_igs_to_adjacent_igs) == 1
      return nothing
    end
    # if at least 3: for now just put everything together
    if length(root_igs_to_adjacent_igs) >= 3
      root_igs = keys(root_igs_to_adjacent_igs)
      root = union(root_igs...)
      igs_to_adjacency_tree[root] = IndexAdjacencyTree(
        [igs_to_adjacency_tree[r] for r in root_igs], false, false
      )
      for r in root_igs
        delete!(igs_to_adjacency_tree, r)
      end
      return nothing
    end
    # if 2: assign adjacent_igs to boundary of root_igs (if possible), then concatenate
    igs1, igs2 = collect(keys(root_igs_to_adjacent_igs))
    reordered_1 = reorder!(
      igs_to_adjacency_tree[igs1], root_igs_to_adjacent_igs[igs1]; boundary="right"
    )
    reordered_2 = reorder!(
      igs_to_adjacency_tree[igs2], root_igs_to_adjacent_igs[igs2]; boundary="left"
    )
    adj_tree_1 = igs_to_adjacency_tree[igs1]
    adj_tree_2 = igs_to_adjacency_tree[igs2]
    if (!reordered_1) && (!reordered_2)
      out_adj_tree = IndexAdjacencyTree([adj_tree_1, adj_tree_2], false, false)
    elseif (!reordered_1)
      out_adj_tree = IndexAdjacencyTree([adj_tree_1, adj_tree_2.children...], false, true)
    elseif (!reordered_2)
      out_adj_tree = IndexAdjacencyTree([adj_tree_1.children..., adj_tree_2], false, true)
    else
      out_adj_tree = IndexAdjacencyTree(
        [adj_tree_1.children..., adj_tree_2.children...], false, true
      )
    end
    root_igs = keys(root_igs_to_adjacent_igs)
    root = union(root_igs...)
    igs_to_adjacency_tree[root] = out_adj_tree
    for r in root_igs
      delete!(igs_to_adjacency_tree, r)
    end
  end
end

# Generate the adjacency tree of a contraction tree
# Args:
# ==========
# ctree: the input contraction tree
# ancestors: ancestor ctrees of the input ctree
# ctree_to_igs: mapping each ctree to neighboring index groups 
function generate_adjacency_tree(ctree, ancestors, ctree_to_igs)
  @timeit timer "generate_adjacency_tree" begin
    # mapping each index group to adjacent input igs
    ig_to_adjacent_igs = Dict{IndexGroup,Set{IndexGroup}}()
    # mapping each igs to an adjacency tree
    igs_to_adjacency_tree = Dict{Set{IndexGroup},IndexAdjacencyTree}()
    for ig in ctree_to_igs[ctree]
      ig_to_adjacent_igs[ig] = Set([ig])
      igs_to_adjacency_tree[Set([ig])] = IndexAdjacencyTree(ig)
    end
    for (i, a) in ancestors
      inter_igs = intersect(ctree_to_igs[a[1]], ctree_to_igs[a[2]])
      new_igs_index = (i == 1) ? 2 : 1
      new_igs = setdiff(ctree_to_igs[a[new_igs_index]], inter_igs)
      # Tensor product is not considered for now
      @assert length(inter_igs) >= 1
      list_adjacent_igs = [ig_to_adjacent_igs[ig] for ig in inter_igs]
      update_igs_to_adjacency_tree!(list_adjacent_igs, igs_to_adjacency_tree)
      for ig in new_igs
        ig_to_adjacent_igs[ig] = union(list_adjacent_igs...)
      end
      if length(igs_to_adjacency_tree) == 1
        return collect(values(igs_to_adjacency_tree))[1]
      end
    end
    if length(igs_to_adjacency_tree) >= 1
      @info "generate_adjacency_tree has ", length(igs_to_adjacency_tree), "outputs"
      return IndexAdjacencyTree([collect(values(igs_to_adjacency_tree))...], false, false)
    end
  end
end

function get_ancestors(ctree)
  @timeit timer "get_ancestors" begin
    ctree_to_ancestors = Dict{Vector,Vector}()
    queue = [ctree]
    ctree_to_ancestors[ctree] = []
    while queue != []
      node = popfirst!(queue)
      if node isa Vector{ITensor}
        continue
      end
      for (i, child) in enumerate(node)
        queue = [queue..., child]
        ctree_to_ancestors[child] = [(i, node), ctree_to_ancestors[node]...]
      end
    end
    return ctree_to_ancestors
  end
end

# Mutates `v` by sorting elements `x[lo:hi]` using the insertion sort algorithm.
# This method is a copy-paste-edit of sort! in base/sort.jl, amended to return the bubblesort distance.
function _insertion_sort(v::Vector, lo::Int, hi::Int)
  @timeit timer "_insertion_sort" begin
    v = copy(v)
    if lo == hi
      return 0
    end
    nswaps = 0
    for i in (lo + 1):hi
      j = i
      x = v[i]
      while j > lo
        if x < v[j - 1]
          nswaps += 1
          v[j] = v[j - 1]
          j -= 1
          continue
        end
        break
      end
      v[j] = x
    end
    return nswaps
  end
end

function insertion_sort(v1::Vector, v2::Vector)
  value_to_index = Dict{Int,Int}()
  for (i, v) in enumerate(v2)
    value_to_index[v] = i
  end
  new_v1 = [value_to_index[v] for v in v1]
  return _insertion_sort(new_v1, 1, length(new_v1))
end

function minswap_adjacency_tree!(adj_tree::IndexAdjacencyTree)
  leaves = Vector{IndexGroup}(get_adj_tree_leaves(adj_tree))
  adj_tree.children = leaves
  adj_tree.fixed_order = true
  return adj_tree.fixed_direction = true
end

function minswap_adjacency_tree!(
  adj_tree::IndexAdjacencyTree, input_tree::IndexAdjacencyTree
)
  nodes = input_tree.children
  node_to_int = Dict{IndexGroup,Int}()
  int_to_node = Dict{Int,IndexGroup}()
  index = 1
  for node in nodes
    node_to_int[node] = index
    int_to_node[index] = node
    index += 1
  end
  for node in topo_sort(adj_tree; type=IndexAdjacencyTree)
    if node.children isa Vector{IndexGroup}
      continue
    end
    children_tree = [get_adj_tree_leaves(n) for n in node.children]
    children_order = vcat(children_tree...)
    input_int_order = [node_to_int[n] for n in nodes if n in children_order]
    if node.fixed_order
      perms = [children_tree, reverse(children_tree)]
    else
      perms = collect(permutations(children_tree))
    end
    nswaps = []
    for perm in perms
      int_order = [node_to_int[n] for n in vcat(perm...)]
      push!(nswaps, insertion_sort(int_order, input_int_order))
    end
    children_tree = perms[argmin(nswaps)]
    node.children = vcat(children_tree...)
    node.fixed_order = true
    node.fixed_direction = true
  end
  int_order = [node_to_int[n] for n in adj_tree.children]
  return _insertion_sort(int_order, 1, length(int_order))
end

function split_igs(igs::Vector{IndexGroup}, inter_igs::Vector{IndexGroup})
  igs_left = Vector{IndexGroup}()
  igs_right = Vector{IndexGroup}()
  target_array = igs_left
  for i in igs
    if i in inter_igs
      target_array = igs_right
      continue
    end
    push!(target_array, i)
  end
  return igs_left, igs_right
end

function minswap_adjacency_tree(
  adj_tree::IndexAdjacencyTree,
  input_tree1::IndexAdjacencyTree,
  input_tree2::IndexAdjacencyTree,
)
  function merge(l1_left, l1_right, l2_left, l2_right)
    if length(l1_left) < length(l2_left)
      left_lists = [[l2_left..., l1_left...]]
    elseif length(l1_left) > length(l2_left)
      left_lists = [[l1_left..., l2_left...]]
    else
      left_lists = [[l2_left..., l1_left...], [l1_left..., l2_left...]]
    end
    if length(l1_right) < length(l2_right)
      right_lists = [[l1_right..., l2_right...]]
    elseif length(l1_right) > length(l2_right)
      right_lists = [[l2_right..., l1_right...]]
    else
      right_lists = [[l2_right..., l1_right...], [l1_right..., l2_right...]]
    end
    out_lists = []
    for l in left_lists
      for r in right_lists
        push!(out_lists, IndexAdjacencyTree([l..., r...], true, true))
      end
    end
    return out_lists
  end
  @timeit timer "minswap_adjacency_tree" begin
    leaves_1 = get_adj_tree_leaves(input_tree1)
    leaves_2 = get_adj_tree_leaves(input_tree2)
    inter_igs = intersect(leaves_1, leaves_2)
    leaves_1_left, leaves_1_right = split_igs(leaves_1, inter_igs)
    leaves_2_left, leaves_2_right = split_igs(leaves_2, inter_igs)
    num_swaps_1 =
      min(length(leaves_1_left), length(leaves_2_left)) +
      min(length(leaves_1_right), length(leaves_2_right))
    num_swaps_2 =
      min(length(leaves_1_left), length(leaves_2_right)) +
      min(length(leaves_1_right), length(leaves_2_left))
    if num_swaps_1 == num_swaps_2
      inputs_1 = merge(leaves_1_left, leaves_1_right, leaves_2_left, leaves_2_right)
      inputs_2 = merge(
        leaves_1_left, leaves_1_right, reverse(leaves_2_right), reverse(leaves_2_left)
      )
      inputs = [inputs_1..., inputs_2...]
    elseif num_swaps_1 > num_swaps_2
      inputs = merge(
        leaves_1_left, leaves_1_right, reverse(leaves_2_right), reverse(leaves_2_left)
      )
    else
      inputs = merge(leaves_1_left, leaves_1_right, leaves_2_left, leaves_2_right)
    end
    # TODO: may want to change this back
    # leaves_1 = [i for i in leaves_1 if !(i in inter_igs)]
    # leaves_2 = [i for i in leaves_2 if !(i in inter_igs)]
    # input1 = IndexAdjacencyTree([leaves_1..., leaves_2...], true, true)
    # input2 = IndexAdjacencyTree([leaves_1..., reverse(leaves_2)...], true, true)
    # input3 = IndexAdjacencyTree([reverse(leaves_1)..., leaves_2...], true, true)
    # input4 = IndexAdjacencyTree([reverse(leaves_1)..., reverse(leaves_2)...], true, true)
    # inputs = [input1, input2, input3, input4]
    # ======================================
    adj_tree_copies = [copy(adj_tree) for _ in 1:length(inputs)]
    nswaps = [minswap_adjacency_tree!(t, i) for (t, i) in zip(adj_tree_copies, inputs)]
    return adj_tree_copies[argmin(nswaps)]
  end
end

function _approximate_contract_pre_process(tn_leaves, ctrees)
  @timeit timer "_approximate_contract_pre_process" begin
    # mapping each contraction tree to its uncontracted index groups
    ctree_to_igs = Dict{Vector,Vector{IndexGroup}}()
    index_groups = get_index_groups(ctrees[end])
    for c in vcat(tn_leaves, ctrees)
      ctree_to_igs[c] = neighbor_index_groups(c, index_groups)
    end
    ctree_to_ancestors = get_ancestors(ctrees[end])
    # mapping each contraction tree to its index adjacency tree
    ctree_to_adj_tree = Dict{Vector,IndexAdjacencyTree}()
    for leaf in tn_leaves
      ctree_to_adj_tree[leaf] = generate_adjacency_tree(
        leaf, ctree_to_ancestors[leaf], ctree_to_igs
      )
      minswap_adjacency_tree!(ctree_to_adj_tree[leaf])
    end
    for c in ctrees
      ancestors = ctree_to_ancestors[c]
      adj_tree = generate_adjacency_tree(c, ancestors, ctree_to_igs)
      if adj_tree != nothing
        ctree_to_adj_tree[c] = minswap_adjacency_tree(
          adj_tree, ctree_to_adj_tree[c[1]], ctree_to_adj_tree[c[2]]
        )
      end
    end
    # mapping each contraction tree to its contract igs
    ctree_to_contract_igs = Dict{Vector,Vector{IndexGroup}}()
    for c in ctrees
      contract_igs = intersect(ctree_to_igs[c[1]], ctree_to_igs[c[2]])
      ctree_to_contract_igs[c[1]] = contract_igs
      ctree_to_contract_igs[c[2]] = contract_igs
    end
    # special case when the network contains uncontracted inds
    ctree_to_contract_igs[ctrees[end]] = ctree_to_igs[ctrees[end]]
    # mapping each index group to a linear ordering
    ig_to_linear_order = Dict{IndexGroup,Vector}()
    for leaf in tn_leaves
      for ig in ctree_to_igs[leaf]
        if !haskey(ig_to_linear_order, ig)
          ig_to_linear_order[ig] = inds_linear_order(leaf, ig.data)
        end
      end
    end
    return ctree_to_igs, ctree_to_adj_tree, ctree_to_contract_igs, ig_to_linear_order
  end
end

function ordered_igs_to_binary_tree(ordered_igs, contract_igs, ig_to_linear_order; ansatz)
  @assert ansatz in ["comb", "mps"]
  @timeit timer "ordered_igs_to_binary_tree" begin
    @assert contract_igs != []
    left_igs, right_igs = split_igs(ordered_igs, contract_igs)
    if ansatz == "comb"
      return ordered_igs_to_binary_tree_comb(
        left_igs, right_igs, contract_igs, ig_to_linear_order
      )
    elseif ansatz == "mps"
      return ordered_igs_to_binary_tree_mps(
        left_igs, right_igs, contract_igs, ig_to_linear_order
      )
    end
  end
end

function ordered_igs_to_binary_tree_mps(
  left_igs, right_igs, contract_igs, ig_to_linear_order
)
  left_order = vcat([ig_to_linear_order[ig] for ig in left_igs]...)
  right_order = vcat([ig_to_linear_order[ig] for ig in right_igs]...)
  contract_order = vcat([ig_to_linear_order[ig] for ig in contract_igs]...)
  if length(left_order) <= length(right_order)
    left_order = [left_order..., contract_order...]
  else
    right_order = [contract_order..., right_order...]
  end
  return merge_tree(line_to_tree(left_order), line_to_tree(reverse(right_order)))
end

function ordered_igs_to_binary_tree_comb(
  left_igs, right_igs, contract_igs, ig_to_linear_order
)
  tree_1 = line_to_tree([line_to_tree(ig_to_linear_order[ig]) for ig in left_igs])
  tree_contract = line_to_tree([
    line_to_tree(ig_to_linear_order[ig]) for ig in contract_igs
  ])
  tree_2 = line_to_tree([line_to_tree(ig_to_linear_order[ig]) for ig in reverse(right_igs)])
  # make the binary tree more balanced to save tree approximation cost
  if tree_1 == []
    return merge_tree(merge_tree(tree_1, tree_contract), tree_2)
  end
  if tree_2 == []
    return merge_tree(tree_1, merge_tree(tree_contract, tree_2))
  end
  if length(vectorize(tree_1)) <= length(vectorize(tree_2))
    return merge_tree(merge_tree(tree_1, tree_contract), tree_2)
  else
    return merge_tree(tree_1, merge_tree(tree_contract, tree_2))
  end
end

function get_igs_cache_info(igs_list, contract_igs_list)
  function split_boundary(list1::Vector{IndexGroup}, list2::Vector{IndexGroup})
    index = 1
    boundary = Vector{IndexGroup}()
    while list1[index] == list2[index]
      push!(boundary, list2[index])
      index += 1
      if index > length(list1) || index > length(list2)
        break
      end
    end
    if index <= length(list1)
      remain_list1 = list1[index:end]
    else
      remain_list1 = Vector{IndexGroup}()
    end
    return boundary, remain_list1
  end
  function split_boundary(igs::Vector{IndexGroup}, lists::Vector{Vector{IndexGroup}})
    if length(igs) <= 1
      return Vector{IndexGroup}(), igs
    end
    for l in lists
      if length(l) >= 2 && igs[1] == l[1] && igs[2] == l[2]
        return split_boundary(igs, l)
      end
    end
    return Vector{IndexGroup}(), igs
  end
  @timeit timer "get_igs_cache_info" begin
    out, input1, input2 = igs_list
    contract_out, contract_input1, contract_input2 = contract_igs_list
    out_left, out_right = split_igs(out, contract_out)
    out_right = reverse(out_right)
    input1_left, input1_right = split_igs(input1, contract_input1)
    input2_left, input2_right = split_igs(input2, contract_input2)
    inputs = [input1_left, reverse(input1_right), input2_left, reverse(input2_right)]
    boundary_left, remain_left = split_boundary(out_left, inputs)
    boundary_right, remain_right = split_boundary(out_right, inputs)
    return [remain_left..., contract_out..., reverse(remain_right)...],
    boundary_left,
    boundary_right
  end
end

function get_tn_cache_sub_info(
  tn_tree::Dict{Vector,OrthogonalITensor}, cache_binary_trees::Vector
)
  cached_tn = []
  cached_tn_tree = Dict{Vector,OrthogonalITensor}()
  new_igs = []
  for binary_tree in cache_binary_trees
    if binary_tree == [] || !haskey(tn_tree, binary_tree)
      push!(new_igs, nothing)
    else
      binary_tree = Vector{Vector}(binary_tree)
      nodes = topo_sort(binary_tree; type=Vector{<:Vector})
      sub_tn = [tn_tree[n] for n in nodes]
      sub_tn_tree = Dict([n => tn_tree[n] for n in nodes]...)
      index_leaves = vectorize(binary_tree)
      new_indices = setdiff(noncommoninds(sub_tn...), index_leaves)
      @assert length(new_indices) == 1
      new_indices = Vector{<:Index}(new_indices)
      push!(new_igs, IndexGroup(new_indices))
      cached_tn = vcat(cached_tn, sub_tn)
      cached_tn_tree = merge(cached_tn_tree, sub_tn_tree)
    end
  end
  tn = vcat(collect(values(tn_tree))...)
  uncached_tn = setdiff(tn, cached_tn)
  return cached_tn_tree, uncached_tn, new_igs
end

function get_tn_cache_info(
  ctree_to_tn_tree::Dict{Vector,Dict{Vector,OrthogonalITensor}},
  ctree_1::Vector,
  ctree_2::Vector,
  cache_binary_trees::Vector,
)
  @timeit timer "get_tn_cache_info" begin
    if haskey(ctree_to_tn_tree, ctree_1)
      tn_tree_1 = ctree_to_tn_tree[ctree_1]
      cached_tn_tree1, uncached_tn1, new_igs_1 = get_tn_cache_sub_info(
        tn_tree_1, cache_binary_trees
      )
    else
      cached_tn_tree1 = Dict{Vector,OrthogonalITensor}()
      uncached_tn1 = get_child_tn(ctree_to_tn_tree, ctree_1)
      new_igs_1 = [nothing, nothing]
    end
    if haskey(ctree_to_tn_tree, ctree_2)
      tn_tree_2 = ctree_to_tn_tree[ctree_2]
      cached_tn_tree2, uncached_tn2, new_igs_2 = get_tn_cache_sub_info(
        tn_tree_2, cache_binary_trees
      )
    else
      cached_tn_tree2 = Dict{Vector,OrthogonalITensor}()
      uncached_tn2 = get_child_tn(ctree_to_tn_tree, ctree_2)
      new_igs_2 = [nothing, nothing]
    end
    uncached_tn = [uncached_tn1..., uncached_tn2...]
    new_igs_left = [i for i in [new_igs_1[1], new_igs_2[1]] if i != nothing]
    @assert length(new_igs_left) <= 1
    if length(new_igs_left) == 1
      new_ig_left = new_igs_left[1]
    else
      new_ig_left = nothing
    end
    new_igs_right = [i for i in [new_igs_1[2], new_igs_2[2]] if i != nothing]
    @assert length(new_igs_right) <= 1
    if length(new_igs_right) == 1
      new_ig_right = new_igs_right[1]
    else
      new_ig_right = nothing
    end
    return merge(cached_tn_tree1, cached_tn_tree2), uncached_tn, new_ig_left, new_ig_right
  end
end

function update_tn_tree_keys!(tn_tree, inds_btree, pairs::Vector{Pair})
  @timeit timer "update_tn_tree_keys!" begin
    current_to_update_key = Dict{Vector,Vector}(pairs...)
    nodes = topo_sort(inds_btree; type=Vector{<:Vector})
    for n in nodes
      @assert haskey(tn_tree, n)
      new_key = n
      if haskey(current_to_update_key, n[1])
        new_key = [current_to_update_key[n[1]], n[2]]
      end
      if haskey(current_to_update_key, n[2])
        new_key = [new_key[1], current_to_update_key[n[2]]]
      end
      if new_key != n
        tn_tree[new_key] = tn_tree[n]
        delete!(tn_tree, n)
        current_to_update_key[n] = new_key
      end
    end
  end
end

function get_child_tn(
  ctree_to_tn_tree::Dict{Vector,Dict{Vector,OrthogonalITensor}}, ctree::Vector
)
  if !haskey(ctree_to_tn_tree, ctree)
    @assert ctree isa Vector{ITensor}
    return orthogonal_tensors(ctree)
  else
    return vcat(collect(values(ctree_to_tn_tree[ctree]))...)
  end
end

# ctree: contraction tree
# tn: vector of tensors representing a tensor network
# tn_tree: a dict maps each index tree in the tn to a tensor
# adj_tree: index adjacency tree
# ig: index group
# contract_ig: the index group to be contracted next
# ig_tree: an index group with a tree hierarchy 
function approximate_contract(ctree::Vector; cutoff, maxdim, ansatz="mps", use_cache=true)
  @timeit timer "approximate_contract" begin
    tn_leaves = get_leaves(ctree)
    ctrees = topo_sort(ctree; leaves=tn_leaves)
    ctree_to_igs, ctree_to_adj_tree, ctree_to_contract_igs, ig_to_linear_order = _approximate_contract_pre_process(
      tn_leaves, ctrees
    )
    # mapping each contraction tree to a tensor network
    ctree_to_tn_tree = Dict{Vector,Dict{Vector,OrthogonalITensor}}()
    # accumulate norm
    log_accumulated_norm = 0.0
    for (ii, c) in enumerate(ctrees)
      @info ii, "th tree approximation"
      if ctree_to_igs[c] == []
        @assert c == ctrees[end]
        tn1 = get_child_tn(ctree_to_tn_tree, c[1])
        tn2 = get_child_tn(ctree_to_tn_tree, c[2])
        tn = vcat(tn1, tn2)
        return get_tensors([optcontract(tn)]), log_accumulated_norm
      end
      # caching is not used here
      if use_cache == false
        tn1 = get_child_tn(ctree_to_tn_tree, c[1])
        tn2 = get_child_tn(ctree_to_tn_tree, c[2])
        inds_btree = ordered_igs_to_binary_tree(
          ctree_to_adj_tree[c].children,
          ctree_to_contract_igs[c],
          ig_to_linear_order;
          ansatz=ansatz,
        )
        ctree_to_tn_tree[c], log_root_norm = approximate_contract_ctree_to_tensor(
          [tn1..., tn2...], inds_btree; cutoff=cutoff, maxdim=maxdim
        )
        log_accumulated_norm += log_root_norm
        continue
      end
      # caching
      # Note: cache_igs_right has a reversed ordering
      center_igs, cache_igs_left, cache_igs_right = get_igs_cache_info(
        [ctree_to_adj_tree[i].children for i in [c, c[1], c[2]]],
        [ctree_to_contract_igs[i] for i in [c, c[1], c[2]]],
      )
      if ansatz == "comb"
        cache_binary_tree_left = line_to_tree([
          line_to_tree(ig_to_linear_order[ig]) for ig in cache_igs_left
        ])
        cache_binary_tree_right = line_to_tree([
          line_to_tree(ig_to_linear_order[ig]) for ig in cache_igs_right
        ])
      elseif ansatz == "mps"
        left_order = vcat([ig_to_linear_order[ig] for ig in cache_igs_left]...)
        cache_binary_tree_left = line_to_tree(left_order)
        right_order = vcat([ig_to_linear_order[ig] for ig in reverse(cache_igs_right)]...)
        cache_binary_tree_right = line_to_tree(reverse(right_order))
      end
      cached_tn_tree, uncached_tn, new_ig_left, new_ig_right = get_tn_cache_info(
        ctree_to_tn_tree, c[1], c[2], [cache_binary_tree_left, cache_binary_tree_right]
      )
      if new_ig_right == nothing && new_ig_left == nothing
        @info "Caching is not used in this approximation"
        @assert length(cached_tn_tree) == 0
      else
        @info "Caching is used in this approximation", new_ig_left, new_ig_right
      end
      new_ig_to_binary_tree_pairs = Vector{Pair}()
      new_igs = center_igs
      new_ig_to_linear_order = ig_to_linear_order
      if new_ig_left == nothing
        new_igs = [cache_igs_left..., new_igs...]
      else
        new_ig_to_linear_order = merge(
          new_ig_to_linear_order, Dict(new_ig_left => [new_ig_left.data])
        )
        new_igs = [new_ig_left, new_igs...]
        push!(new_ig_to_binary_tree_pairs, new_ig_left.data => cache_binary_tree_left)
      end
      if new_ig_right == nothing
        new_igs = [new_igs..., cache_igs_right...]
      else
        new_ig_to_linear_order = merge(
          new_ig_to_linear_order, Dict(new_ig_right => [new_ig_right.data])
        )
        new_igs = [new_igs..., new_ig_right]
        push!(new_ig_to_binary_tree_pairs, new_ig_right.data => cache_binary_tree_right)
      end
      inds_btree = ordered_igs_to_binary_tree(
        new_igs, ctree_to_contract_igs[c], new_ig_to_linear_order; ansatz=ansatz
      )
      new_tn_tree, log_root_norm = approximate_contract_ctree_to_tensor(
        uncached_tn, inds_btree; cutoff=cutoff, maxdim=maxdim
      )
      log_accumulated_norm += log_root_norm
      if length(new_ig_to_binary_tree_pairs) != 0
        update_tn_tree_keys!(new_tn_tree, inds_btree, new_ig_to_binary_tree_pairs)
      end
      ctree_to_tn_tree[c] = merge(new_tn_tree, cached_tn_tree)
    end
    tn = vcat(collect(values(ctree_to_tn_tree[ctrees[end]]))...)
    return get_tensors(tn), log_accumulated_norm
  end
end

# interlaced HOSVD using caching
function tree_approximation_cache(
  embedding::Dict, inds_btree::Vector; cutoff=1e-15, maxdim=10000, maxsize=10000
)
  @info "start tree_approximation_cache", inds_btree
  ctree_to_tensor = Dict{Vector,OrthogonalITensor}()
  # initialize sim_dict
  network = vcat(collect(values(embedding))...)
  uncontractinds = noncommoninds(network...)
  innerinds = mapreduce(t -> [i for i in inds(t)], vcat, network)
  innerinds = Vector(setdiff(innerinds, uncontractinds))
  siminner_dict = Dict([ind => sim(ind) for ind in innerinds])

  function closednet(tree)
    netbra = embedding[tree]
    netket = replaceinds(netbra, siminner_dict)
    if length(tree) == 1
      return optcontract(vcat(netbra, netket))
    end
    tleft, tright = closednet(tree[1]), closednet(tree[2])
    return optcontract(vcat(netbra, netket, [tleft], [tright]))
  end

  function insert_projectors(tree::Vector, env::OrthogonalITensor)
    netbra = embedding[tree]
    netket = replaceinds(netbra, siminner_dict)
    if length(tree) == 1
      tensor_bra = optcontract(netbra)
      tensor_ket = replaceinds([tensor_bra], siminner_dict)[1]
      inds_pair = (tree[1], sim(tree[1]))
      tensor_ket = replaceinds([tensor_ket], Dict([inds_pair[1] => inds_pair[2]]))[1]
      return inds_pair, optcontract([netbra..., netket...]), [tensor_bra, tensor_ket]
    end
    # update children
    subenvtensor = optcontract([env, netbra...])
    envnet = [subenvtensor, closednet(tree[2]), netket...]
    ind1_pair, subnetsq1, subnet1 = insert_projectors(tree[1], optcontract(envnet))
    envnet = [subenvtensor, subnetsq1, netket...]
    ind2_pair, _, subnet2 = insert_projectors(tree[2], optcontract(envnet))
    # compute the projector
    rinds = (ind1_pair[1], ind2_pair[1])
    linds = (ind1_pair[2], ind2_pair[2])
    # to handle the corner cases where subnet1/subnet2 could be empty
    netket = replaceinds(
      netket, Dict([ind1_pair[1] => ind1_pair[2], ind2_pair[1] => ind2_pair[2]])
    )
    net = [subenvtensor, netket..., subnet1..., subnet2...]
    tnormal = optcontract(net)
    dim2 = floor(maxsize / (space(ind1_pair[1]) * space(ind2_pair[1])))
    dim = min(maxdim, dim2)
    t00 = time()
    @info "eigen input size", size(tnormal.tensor)
    @timeit timer "eigen" begin
      diag, U = eigen(
        tnormal.tensor, linds, rinds; cutoff=cutoff, maxdim=dim, ishermitian=true
      )
    end
    t11 = time() - t00
    @info "size of U", size(U), "size of diag", size(diag), "costs", t11
    dr = commonind(diag, U)
    Usim = replaceinds(U, rinds => linds)
    ortho_U = OrthogonalITensor(U)
    ortho_Usim = OrthogonalITensor(Usim)
    net1 = [netbra..., subnet1[1], subnet2[1], ortho_U]
    net2 = [netket..., subnet1[2], subnet2[2], ortho_Usim]
    tensor1 = optcontract(net1)
    tensor2 = replaceinds(tensor1, noncommoninds(net1...) => noncommoninds(net2...))
    subnetsq = optcontract([tensor1, tensor2])
    dr_pair = (dr, sim(dr))
    tensor2 = replaceinds(tensor2, [dr_pair[1]] => [dr_pair[2]])
    subnet = [tensor1, tensor2]
    ctree_to_tensor[tree] = ortho_U
    return dr_pair, subnetsq, subnet
  end

  @assert (length(inds_btree) >= 2)
  bra = embedding[inds_btree]
  ket = replaceinds(bra, siminner_dict)
  # update children
  envnet = [closednet(inds_btree[2]), bra..., ket...]
  _, netsq1, n1 = insert_projectors(inds_btree[1], optcontract(envnet))
  envnet = [netsq1, bra..., ket...]
  _, _, n2 = insert_projectors(inds_btree[2], optcontract(envnet))
  # last tensor
  envnet = [n1[1], n2[1], bra...]
  last_tensor = optcontract(envnet)
  root_norm = norm(last_tensor.tensor)
  last_tensor.tensor /= root_norm
  ctree_to_tensor[inds_btree] = last_tensor
  return ctree_to_tensor, log(root_norm)
end
