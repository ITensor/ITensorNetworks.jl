function build_tntree_unbalanced(
  tn::ITensorNetwork; nvertices_per_partition=2, backend="KaHyPar"
)
  @assert is_connected(tn)
  g_parts = partition(tn; nvertices_per_partition=nvertices_per_partition, backend=backend)
  @assert is_connected(g_parts)
  root = 1
  tree = bfs_tree(g_parts, root)
  tntree = nothing
  queue = [root]
  while queue != []
    v = popfirst!(queue)
    queue = vcat(queue, child_vertices(tree, v))
    if tntree == nothing
      tntree = Vector{ITensor}(g_parts[v])
    else
      tntree = [tntree, Vector{ITensor}(g_parts[v])]
    end
  end
  return tntree
end

function build_tntree_balanced(
  tn::ITensorNetwork; nvertices_per_partition=2, backend="KaHyPar"
)
  # @assert is_connected(tn)
  g_parts = partition(tn; npartitions=2, backend=backend)
  if nv(g_parts[1]) >= max(nvertices_per_partition, 2)
    tntree_1 = build_tntree_balanced(g_parts[1]; nvertices_per_partition, backend)
  else
    tntree_1 = Vector{ITensor}(g_parts[1])
  end
  if nv(g_parts[2]) >= max(nvertices_per_partition, 2)
    tntree_2 = build_tntree_balanced(g_parts[2]; nvertices_per_partition, backend=backend)
  else
    tntree_2 = Vector{ITensor}(g_parts[2])
  end
  return [tntree_1, tntree_2]
end
