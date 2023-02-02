
using ITensors
using ITensorNetworks
using NamedGraphs: add_vertex!, add_edge!

struct TupleTree
  num::Tuple
  children::Vector{TupleTree}
end

function create_contraction_tree(contract_sequence)
  verts = collect(Leaves(contract_sequence))
  g = DataGraph()
  spawn_child_branch(contract_sequence[1], g, verts)
  spawn_child_branch(contract_sequence[2], g, verts)

  #Now for the edges!
  for v in vertices(g)
    if (length(v) == 3)
      #Figure out its children, it will be such that the concatenation of two of its branches are a branch
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

function spawn_child_branch(contract_sequence, current_g, vertices_full)
  if (length(seq) != 1)
    group1 = if typeof(contract_sequence[1]) == vertex_data_type
      [contract_sequence[1]]
    else
      collect(Leaves(contract_sequence[1]))
    end
    group2 = if typeof(contract_sequence[2]) == vertex_data_type
      [contract_sequence[2]]
    else
      collect(Leaves(contract_sequence[2]))
    end
    remaining_verts = setdiff(vertices_full, vcat(group1, group2))
    add_vertex!(current_g, (group1, group2, remaining_verts))
    spawn_child_branch(contract_sequence, current_g, vertices_full)
    spawn_child_branch(contract_sequence, current_g, vertices_full)
  else
    add_vertex!(
      current_g, ([contract_sequence], setdiff(vertices_full, [contract_sequence]))
    )
  end
end

function contraction_node(g::DataGraph, v)
  return length(neighbors(g, v)) != 1
end

function contraction_edge(g::DataGraph, e)
  return contraction_node(g, src(e)) && contraction_node(g, dst(e))
end

function internal_contraction_node(g::DataGraph, v)
  for vn in neighbors(g, v)
    if (!contraction_node(g, vn))
      return false
    end
  end
  return true
end

function contraction_edges(g)
  return edges(g)[findall(==(1), [contraction_edge(g, e) for e in edges(g)])]
end

function internal_contraction_nodes(g)
  return vertices(g)[findall(
    ==(1), [internal_contraction_node(g, v) && contraction_node(g, v) for v in vertices(g)]
  )]
end

function external_contraction_nodes(g)
  return vertices(g)[findall(
    ==(1), [!internal_contraction_node(g, v) && contraction_node(g, v) for v in vertices(g)]
  )]
end

function edge_bipartition(g::DataGraph, e)
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

  return left_bipartition, right_bipartition
end

function external_contraction_node_int_tensors(g::DataGraph, v)
  return [Base.Iterators.flatten(v[findall(==(1), [length(vi) == 1 for vi in v])])...]
end

function external_contraction_node_ext_tensors(g::DataGraph, v)
  return [Base.Iterators.flatten(v[findall(==(1), [length(vi) != 1 for vi in v])])...]
end
