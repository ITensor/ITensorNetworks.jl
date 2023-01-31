
"""
Take a contraction_sequence and return a graphical representation of it. The leaves of the graph represent the leaves of the sequence whilst the internal_nodes of the graph
define a tripartition of the graph and thus are named as an n = 3 element tuples, which each element specifying the keys involved.
Edges connect parents/children within the contraction sequence.
"""
function contraction_sequence_to_graph(contract_sequence)
  g = fill_contraction_sequence_graph_vertices(contract_sequence)

  #Now we have the vertices we need to figure out the edges
  for v in vertices(g)
    #Only add edges from a parent (which defines a tripartition and thus has length 3) to its children
    if (length(v) == 3)
      #Work out which vertices it connects to
      concat1, concat2, concat3 = [v[1]..., v[2]...], [v[2]..., v[3]...], [v[1]..., v[3]...]
      for vn in setdiff(vertices(g), [v])
        vn_set = [Set(vni) for vni in vn]
        if (Set(concat1) ∈ vn_set || Set(concat2) ∈ vn_set || Set(concat3) ∈ vn_set)
          add_edge!(g, v => vn)
        end
      end
    end
  end

  return g
end

function fill_contraction_sequence_graph_vertices(contract_sequence)
  g = NamedGraph()
  leaves = collect(Leaves(contract_sequence))
  fill_contraction_sequence_graph_vertices!(g, contract_sequence[1], leaves)
  fill_contraction_sequence_graph_vertices!(g, contract_sequence[2], leaves)
  return g
end

"""Given a contraction sequence which is a subsequence of some larger sequence (with leaves `leaves`) which is being built on g
Spawn `contract sequence' as a vertex on `current_g' and continue on with its children """
function fill_contraction_sequence_graph_vertices!(g, contract_sequence, leaves)
  if (isa(contract_sequence, Array))
    group1 = collect(Leaves(contract_sequence[1]))
    group2 = collect(Leaves(contract_sequence[2]))
    remaining_verts = setdiff(leaves, vcat(group1, group2))
    add_vertex!(g, (group1, group2, remaining_verts))
    fill_contraction_sequence_graph_vertices!(g, contract_sequence[1], leaves)
    fill_contraction_sequence_graph_vertices!(g, contract_sequence[2], leaves)
  else
    add_vertex!(g, ([contract_sequence], setdiff(leaves, [contract_sequence])))
  end
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
