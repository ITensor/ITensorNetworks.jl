@eval module $(gensym())
using Graphs: vertices
using ITensorNetworks:
  contraction_sequence,
  contraction_sequence_to_digraph,
  contraction_sequence_to_graph,
  contraction_tree_leaf_bipartition,
  random_tensornetwork
using NamedGraphs.GraphsExtensions:
  is_leaf_vertex, leaf_vertices, non_leaf_edges, root_vertex
using NamedGraphs.NamedGraphGenerators: named_grid
using StableRNGs: StableRNG
using Test: @test, @testset
@testset "contraction_sequence_to_graph" begin
  n = 3
  dims = (n, n)
  g = named_grid(dims)
  rng = StableRNG(1234)
  tn = random_tensornetwork(rng, g; link_space=2)
  seq = contraction_sequence(tn)
  g_directed_seq = contraction_sequence_to_digraph(seq)
  g_seq_leaves = leaf_vertices(g_directed_seq)
  @test length(g_seq_leaves) == n * n
  @test 2 * length(g_seq_leaves) - 1 == length(vertices(g_directed_seq))
  @test root_vertex(g_directed_seq)[3] == []

  g_seq = contraction_sequence_to_graph(seq)
  @test length(g_seq_leaves) == n * n
  @test 2 * length(g_seq_leaves) - 2 == length(vertices(g_seq))

  for eb in non_leaf_edges(g_seq)
    vs = contraction_tree_leaf_bipartition(g_seq, eb)
    @test length(vs) == 2
    @test Set([v.I for v in vcat(vs[1], vs[2])]) == Set(vertices(tn))
  end
  #Check all internal vertices define a correct tripartition and all leaf vertices define a bipartition (tensor on that leafs vs tensor on rest of tree)
  for v in vertices(g_seq)
    if (!is_leaf_vertex(g_seq, v))
      @test length(v) == 3
      @test Set([vsi.I for vsi in vcat(v[1], v[2], v[3])]) == Set(vertices(tn))
    else
      @test length(v) == 2
      @test Set([vsi.I for vsi in vcat(v[1], v[2])]) == Set(vertices(tn))
    end
  end
end
end
