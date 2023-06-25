"""
Take a contraction sequence and return a directed graph.
"""
function contraction_sequence_to_digraph(contract_sequence)
  g = NamedDiGraph()
  leaves = collect(Leaves(contract_sequence))
  seq_to_v = Dict()
  for seq in PostOrderDFS(contract_sequence)
    if !(seq isa Array)
      v = ([seq], setdiff(leaves, [seq]))
      add_vertex!(g, v)
    else
      group1 = collect(Leaves(seq[1]))
      group2 = collect(Leaves(seq[2]))
      remaining_verts = setdiff(leaves, vcat(group1, group2))
      v = (group1, group2, remaining_verts)
      add_vertex!(g, v)
      c1 = get(seq_to_v, seq[1], nothing)
      c2 = get(seq_to_v, seq[2], nothing)
      add_edge!(g, v => c1)
      add_edge!(g, v => c2)
    end
    seq_to_v[seq] = v
  end
  return g
end

"""
Take a contraction_sequence and return a graphical representation of it. The leaves of the graph represent the leaves of the sequence whilst the internal_nodes of the graph
define a tripartition of the graph and thus are named as an n = 3 element tuples, which each element specifying the keys involved.
Edges connect parents/children within the contraction sequence.
"""
function contraction_sequence_to_graph(contract_sequence)
  direct_g = contraction_sequence_to_digraph(contract_sequence)
  g = NamedGraph(vertices(direct_g))
  for e in edges(direct_g)
    add_edge!(g, e)
  end
  root = _root(direct_g)
  c1, c2 = child_vertices(direct_g, root)
  rem_vertex!(g, root)
  add_edge!(g, c1 => c2)
  return g
end

"""Get the vertex bi-partition that a given edge represents"""
function contraction_tree_leaf_bipartition(g::AbstractGraph, e)
  if (!is_leaf_edge(g, e))
    vsrc_set, vdst_set = [Set(vni) for vni in src(e)], [Set(vni) for vni in dst(e)]
    c1, c2, c3 = [src(e)[1]..., src(e)[2]...],
    [src(e)[2]..., src(e)[3]...],
    [src(e)[1]..., src(e)[3]...]
    left_bipartition = if Set(c1) ∈ vdst_set
      c1
    elseif Set(c2) ∈ vdst_set
      c2
    else
      c3
    end

    c1, c2, c3 = [dst(e)[1]..., dst(e)[2]...],
    [dst(e)[2]..., dst(e)[3]...],
    [dst(e)[1]..., dst(e)[3]...]
    right_bipartition = if Set(c1) ∈ vsrc_set
      c1
    elseif Set(c2) ∈ vsrc_set
      c2
    else
      c3
    end
  else
    left_bipartition = filter(vs -> Set(vs) ∈ [Set(vni) for vni in dst(e)], src(e))[1]
    right_bipartition = setdiff(src(e), left_bipartition)
  end

  return left_bipartition, right_bipartition
end
