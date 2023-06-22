using ITensorNetworks
using ITensorNetworks:
  contraction_sequence_to_directed_graph,
  contraction_sequence_to_graph,
  internal_edges,
  contraction_tree_leaf_bipartition,
  distance_to_leaf,
  leaf_vertices,
  _root
using Test
using ITensors
using NamedGraphs

@testset "contraction_sequence_to_graph" begin
  n = 3
  dims = (n, n)
  g = named_grid(dims)
  s = siteinds("S=1/2", g)

  ψ = randomITensorNetwork(s; link_space=2)
  ψψ = flatten_networks(ψ, ψ)

  seq = contraction_sequence(ψψ)

  g_directed_seq = contraction_sequence_to_directed_graph(seq)
  g_seq_leaves = leaf_vertices(g_directed_seq)
  @test length(g_seq_leaves) == n * n
  @test 2 * length(g_seq_leaves) - 1 == length(vertices(g_directed_seq))
  @test _root(g_directed_seq)[3] == []

  g_seq = contraction_sequence_to_graph(seq)
  @test length(g_seq_leaves) == n * n
  @test 2 * length(g_seq_leaves) - 2 == length(vertices(g_seq))

  for eb in internal_edges(g_seq)
    vs = contraction_tree_leaf_bipartition(g_seq, eb)
    @test length(vs) == 2
    @test Set([v.I for v in vcat(vs[1], vs[2])]) == Set(vertices(ψψ))
  end
  #Check all internal vertices define a correct tripartition and all leaf vertices define a bipartition (tensor on that leafs vs tensor on rest of tree)
  for v in vertices(g_seq)
    if (!is_leaf(g_seq, v))
      @test length(v) == 3
      @test Set([vsi.I for vsi in vcat(v[1], v[2], v[3])]) == Set(vertices(ψψ))
    else
      @test length(v) == 2
      @test Set([vsi.I for vsi in vcat(v[1], v[2])]) == Set(vertices(ψψ))
    end
  end
end
