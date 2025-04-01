using DataStructures: DataStructures, DisjointSets, find_root!
## using ITensors.NDTensors: ind
using .ITensorsExtensions: is_delta

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
Given a list of delta tensors `deltas`, return a `DisjointSets` of all its indices
such that each pair of indices adjacent to any delta tensor must be in the same disjoint set.
If a disjoint set contains indices in `rootinds`, then one of such indices in `rootinds`
must be the root of this set.
"""
function _delta_inds_disjointsets(deltas::Vector{<:ITensor}, rootinds::Vector{<:Index})
  if deltas == []
    return DisjointSets()
  end
  inds_list = map(t -> collect(inds(t)), deltas)
  deltainds = collect(Set(vcat(inds_list...)))
  ds = DisjointSets(deltainds)
  for t in deltas
    i1, i2 = inds(t)
    if find_root!(ds, i1) in rootinds
      _root_union!(ds, find_root!(ds, i1), find_root!(ds, i2))
    else
      _root_union!(ds, find_root!(ds, i2), find_root!(ds, i1))
    end
  end
  return ds
end

"""
Given an input tensor network `tn`, remove redundent delta tensors
in `tn` and change inds accordingly to make the output `tn` represent
the same tensor network but with less delta tensors.

========
Example:
  julia> is = [Index(2, string(i)) for i in 1:6]
  julia> a = ITensor(is[1], is[2])
  julia> b = ITensor(is[2], is[3])
  julia> delta1 = delta(is[3], is[4])
  julia> delta2 = delta(is[5], is[6])
  julia> tn = ITensorNetwork([a, b, delta1, delta2])
  julia> ITensorNetworks._contract_deltas(tn)
  ITensorNetwork{Int64} with 3 vertices:
  3-element Vector{Int64}:
   1
   2
   4

  and 1 edge(s):
  1 => 2

  with vertex data:
  3-element Dictionaries.Dictionary{Int64, Any}
   1 │ ((dim=2|id=457|"1"), (dim=2|id=296|"2"))
   2 │ ((dim=2|id=296|"2"), (dim=2|id=613|"4"))
   4 │ ((dim=2|id=626|"6"), (dim=2|id=237|"5"))
"""
function _contract_deltas(tn::ITensorNetwork)
  deltas = filter(is_delta, collect(eachtensor(tn)))
  if isempty(deltas)
    return tn
  end
  tn = copy(tn)
  outinds = flatten_siteinds(tn)
  ds = _delta_inds_disjointsets(deltas, outinds)
  deltainds = [ds...]
  sim_deltainds = [find_root!(ds, i) for i in deltainds]
  # `rem_vertex!(tn, v)` changes `vertices(tn)` in place.
  # We copy it here so that the enumeration won't be affected.
  vs = copy(vertices(tn))
  for v in vs
    if !is_delta(tn[v])
      tn[v] = replaceinds(tn[v], deltainds, sim_deltainds)
      continue
    end
    i1, i2 = inds(tn[v])
    root = find_root!(ds, i1)
    @assert root == find_root!(ds, i2)
    if i1 != root && i1 in outinds
      tn[v] = delta(i1, root)
    elseif i2 != root && i2 in outinds
      tn[v] = delta(i2, root)
    else
      rem_vertex!(tn, v)
    end
  end
  return tn
end

"""
Given an input `partition`, contract redundent delta tensors of non-leaf vertices
in `partition` without changing the tensor network value.
`root` is the root of the dfs_tree that defines the leaves.
Note: for each vertex `v` of `partition`, the number of non-delta tensors
  in `partition[v]` will not be changed.
Note: only delta tensors of non-leaf vertices will be contracted.
Note: this function assumes that all noncommoninds of the partition are in leaf partitions.
"""
function _contract_deltas_ignore_leaf_partitions(
  partition::DataGraph; root=first(vertices(partition))
)
  partition = copy(partition)
  leaves = leaf_vertices(dfs_tree(partition, root))
  nonleaves = setdiff(vertices(partition), leaves)
  rootinds = _noncommoninds(subgraph(partition, nonleaves))
  # check rootinds are not noncommoninds of the partition
  @assert isempty(intersect(rootinds, _noncommoninds(partition)))
  nonleaves_tn = _contract_deltas(reduce(union, [partition[v] for v in nonleaves]))
  nondelta_vs = filter(v -> !is_delta(nonleaves_tn[v]), vertices(nonleaves_tn))
  for v in nonleaves
    partition[v] = subgraph(nonleaves_tn, intersect(nondelta_vs, vertices(partition[v])))
  end
  # Note: we also need to change inds in the leaves since they can be connected by deltas
  # in nonleaf vertices
  delta_vs = setdiff(vertices(nonleaves_tn), nondelta_vs)
  if isempty(delta_vs)
    return partition
  end
  ds = _delta_inds_disjointsets(
    Vector{ITensor}(subgraph(nonleaves_tn, delta_vs)), Vector{Index}()
  )
  deltainds = Index[ds...]
  sim_deltainds = Index[find_root!(ds, ind) for ind in deltainds]
  for tn_v in leaves
    partition[tn_v] = map_data(partition[tn_v]; edges=[]) do t
      return replaceinds(t, deltainds, sim_deltainds)
    end
  end
  return partition
end
