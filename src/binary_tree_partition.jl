"""
Rewrite of the function
  `DataStructures.root_union!(s::IntDisjointSet{T}, x::T, y::T) where {T<:Integer}`.
"""
function _introot_union!(s::DataStructures.IntDisjointSets, x, y; left_root=true)
  parents = s.parents
  rks = s.ranks
  @inbounds xrank = rks[x]
  @inbounds yrank = rks[y]
  if !left_root
    x, y = y, x
  end
  @inbounds parents[y] = x
  s.ngroups -= 1
  return x
end

"""
Rewrite of the function `DataStructures.root_union!(s::DisjointSet{T}, x::T, y::T)`.
The difference is that in the output of `_root_union!`, x is guaranteed to be the root of y when
setting `left_root=true`, and y will be the root of x when setting `left_root=false`.
In `DataStructures.root_union!`, the root value cannot be specified.
A specified root is useful in functions such as `_remove_deltas`, where when we union two
indices into one disjointset, we want the index that is the outinds if the given tensor network
to always be the root in the DisjointSets.
"""
function _root_union!(s::DisjointSets, x, y; left_root=true)
  return s.revmap[_introot_union!(s.internal, s.intmap[x], s.intmap[y]; left_root=true)]
end

"""
Partition the input network containing both `tn` and `deltas` (a vector of delta tensors)
into two partitions, one adjacent to source_inds and the other adjacent to other external
inds of the network.
"""
function _binary_partition(
  tn::ITensorNetwork, deltas::Vector{ITensor}, source_inds::Vector{<:Index}
)
  all_tensors = [Vector{ITensor}(tn)..., deltas...]
  external_inds = noncommoninds(all_tensors...)
  # add delta tensor to each external ind
  external_sim_ind = [sim(ind) for ind in external_inds]
  new_deltas = [
    delta(external_inds[i], external_sim_ind[i]) for i in 1:length(external_inds)
  ]
  deltas = map(t -> replaceinds(t, external_inds => external_sim_ind), deltas)
  deltas = [deltas..., new_deltas...]
  tn = map_data(t -> replaceinds(t, external_inds => external_sim_ind), tn; edges=[])
  p1, p2 = _mincut_partition_maxweightoutinds(
    disjoint_union(tn, ITensorNetwork(deltas)),
    source_inds,
    setdiff(external_inds, source_inds),
  )
  tn_vs = [v[1] for v in p1 if v[2] == 1]
  source_tn = subgraph(tn, tn_vs)
  delta_indices = [v[1] for v in p1 if v[2] == 2]
  source_deltas = Vector{ITensor}([deltas[i] for i in delta_indices])
  source_tn, source_deltas = _remove_deltas(source_tn, source_deltas)
  tn_vs = [v[1] for v in p2 if v[2] == 1]
  remain_tn = subgraph(tn, tn_vs)
  delta_indices = [v[1] for v in p2 if v[2] == 2]
  remain_deltas = Vector{ITensor}([deltas[i] for i in delta_indices])
  remain_tn, remain_deltas = _remove_deltas(remain_tn, remain_deltas)
  @assert (
    length(noncommoninds(all_tensors...)) == length(
      noncommoninds(
        Vector{ITensor}(source_tn)...,
        source_deltas...,
        Vector{ITensor}(remain_tn)...,
        remain_deltas...,
      ),
    )
  )
  return source_tn, source_deltas, remain_tn, remain_deltas
end

"""
Given an input tensor network containing tensors in the input `tn` and
tensors in `deltas`, remove redundent delta tensors in `deltas` and change
inds accordingly to make the output `tn` and `out_deltas` represent the same
tensor network but with less delta tensors.
Note: inds of tensors in `tn` and `deltas` may be changed, and `out_deltas`
  may still contain necessary delta tensors.

========
Example:
  julia> is = [Index(2, "i") for i in 1:6]
  julia> a = ITensor(is[1], is[2])
  julia> b = ITensor(is[2], is[3])
  julia> delta1 = delta(is[3], is[4])
  julia> delta2 = delta(is[5], is[6])
  julia> tn = ITensorNetwork([a,b])
  julia> tn, out_deltas = ITensorNetworks._remove_deltas(tn, [delta1, delta2])
  julia> noncommoninds(Vector{ITensor}(tn)...)
  2-element Vector{Index{Int64}}:
   (dim=2|id=339|"1")
   (dim=2|id=489|"4")
  julia> length(out_deltas)
  1
"""
function _remove_deltas(tn::ITensorNetwork, deltas::Vector{ITensor})
  out_delta_inds = Vector{Pair}()
  network = [Vector{ITensor}(tn)..., deltas...]
  outinds = noncommoninds(network...)
  inds_list = map(t -> collect(inds(t)), deltas)
  deltainds = collect(Set(vcat(inds_list...)))
  ds = DisjointSets(deltainds)
  for t in deltas
    i1, i2 = inds(t)
    if find_root!(ds, i1) in outinds && find_root!(ds, i2) in outinds
      push!(out_delta_inds, find_root!(ds, i1) => find_root!(ds, i2))
    end
    if find_root!(ds, i1) in outinds
      _root_union!(ds, find_root!(ds, i1), find_root!(ds, i2))
    else
      _root_union!(ds, find_root!(ds, i2), find_root!(ds, i1))
    end
  end
  sim_deltainds = [find_root!(ds, i) for i in deltainds]
  tn = map_data(t -> replaceinds(t, deltainds => sim_deltainds), tn; edges=[])
  out_deltas = Vector{ITensor}([delta(i.first, i.second) for i in out_delta_inds])
  return tn, out_deltas
end

"""
Given an input tn and a rooted binary tree of indices, return a partition of tn with the
same binary tree structure as inds_btree.
Note: in the output partition, we add multiple delta tensors to the network so that
  the output graph is guaranteed to be the same binary tree as inds_btree.
Note: in the output partition, tensor vertex names will be changed. For a given input
  tensor with vertex name `v``, its name in the output partition will be `(v, 1)`, and any
  delta tensor will have name `(v, 2)`.
Note: for a given binary tree with n indices, the output partition will contain 2n-1 vertices,
  with each leaf vertex corresponding to a sub tn adjacent to one output index. Keeping these
  leaf vertices in the partition makes later `approx_itensornetwork` algorithms more efficient.
"""
function binary_tree_partition(tn::ITensorNetwork, inds_btree::Vector)
  output_tns = Vector{ITensorNetwork}()
  output_deltas_vector = Vector{Vector{ITensor}}()
  # Mapping each vertex of the binary tree to a tn and a vector of deltas
  # representing the partition of the subtree containing this vertex and
  # its descendant vertices.
  v_to_subtree_tn_deltas = Dict{Union{Vector,Index},Tuple}()
  v_to_subtree_tn_deltas[inds_btree] = (tn, Vector{ITensor}())
  for v in PreOrderDFS(inds_btree)
    @assert haskey(v_to_subtree_tn_deltas, v)
    input_tn, input_deltas = v_to_subtree_tn_deltas[v]
    if v isa Index
      push!(output_tns, input_tn)
      push!(output_deltas_vector, input_deltas)
      continue
    end
    tn1, deltas1, input_tn, input_deltas = _binary_partition(
      input_tn, input_deltas, collect(Leaves(v[1]))
    )
    v_to_subtree_tn_deltas[v[1]] = (tn1, deltas1)
    tn1, deltas1, input_tn, input_deltas = _binary_partition(
      input_tn, input_deltas, collect(Leaves(v[2]))
    )
    v_to_subtree_tn_deltas[v[2]] = (tn1, deltas1)
    push!(output_tns, input_tn)
    push!(output_deltas_vector, input_deltas)
  end
  # In subgraph_vertices, each element is a vector of vertices to be
  # grouped in one partition.
  subgraph_vs = Vector{Vector{Tuple}}()
  delta_num = 0
  for (tn, deltas) in zip(output_tns, output_deltas_vector)
    vs = Vector{Tuple}([(v, 1) for v in vertices(tn)])
    vs = vcat(vs, [(i + delta_num, 2) for i in 1:length(deltas)])
    push!(subgraph_vs, vs)
    delta_num += length(deltas)
  end
  out_tn = ITensorNetwork()
  for tn in output_tns
    for v in vertices(tn)
      add_vertex!(out_tn, v)
      out_tn[v] = tn[v]
    end
  end
  tn_deltas = ITensorNetwork(vcat(output_deltas_vector...))
  return partition(ITensorNetwork{Any}(disjoint_union(out_tn, tn_deltas)), subgraph_vs)
end
