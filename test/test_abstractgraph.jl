using Test
using Graphs: binary_tree
using ITensorNetworks
using Random

@testset "Number of neighbors" begin
  g1 = comb_tree((3, 2))
  @test num_neighbors(g1, 4) == 1
  @test num_neighbors(g1, 3) == 2
  @test num_neighbors(g1, 2) == 3

  ng1 = named_comb_tree((3, 2))
  @test num_neighbors(ng1, 1, 2) == 1
  @test num_neighbors(ng1, (3, 1)) == 2
  @test num_neighbors(ng1, (2, 1)) == 3

  g2 = grid((3, 3))
  @test num_neighbors(g2, 1) == 2
  @test num_neighbors(g2, 2) == 3
  @test num_neighbors(g2, 5) == 4

  ng2 = named_grid((3, 3))
  @test num_neighbors(ng2, 1, 1) == 2
  @test num_neighbors(ng2, (1, 2)) == 3
  @test num_neighbors(ng2, (2, 2)) == 4
end

# TODO: remove once this is merged into NamedGraphs.jl
@testset "Tree graph paths" begin
  g1 = comb_tree((3, 2))
  et1 = edgetype(g1)
  @test vertex_path(g1, 4, 5) == [4, 1, 2, 5]
  @test edge_path(g1, 4, 5) == [et1(4, 1), et1(1, 2), et1(2, 5)]
  @test vertex_path(g1, 6, 1) == [6, 3, 2, 1]
  @test edge_path(g1, 6, 1) == [et1(6, 3), et1(3, 2), et1(2, 1)]
  @test vertex_path(g1, 2, 2) == [2]
  @test edge_path(g1, 2, 2) == et1[]

  ng1 = named_comb_tree((3, 2))
  net1 = edgetype(ng1)
  @test vertex_path(ng1, (1, 2), (2, 2)) == [(1, 2), (1, 1), (2, 1), (2, 2)]
  @test edge_path(ng1, (1, 2), (2, 2)) ==
    [net1((1, 2), (1, 1)), net1((1, 1), (2, 1)), net1((2, 1), (2, 2))]
  @test vertex_path(ng1, (3, 2), (1, 1)) == [(3, 2), (3, 1), (2, 1), (1, 1)]
  @test edge_path(ng1, (3, 2), (1, 1)) ==
    [net1((3, 2), (3, 1)), net1((3, 1), (2, 1)), net1((2, 1), (1, 1))]
  @test vertex_path(ng1, (1, 2), (1, 2)) == [(1, 2)]
  @test edge_path(ng1, (1, 2), (1, 2)) == net1[]

  g2 = binary_tree(3)
  et2 = edgetype(g2)
  @test vertex_path(g2, 2, 6) == [2, 1, 3, 6]
  @test edge_path(g2, 2, 6) == [et2(2, 1), et2(1, 3), et2(3, 6)]
  @test vertex_path(g2, 5, 4) == [5, 2, 4]
  @test edge_path(g2, 5, 4) == [et2(5, 2), et2(2, 4)]

  ng2 = named_binary_tree(3)
  net2 = edgetype(ng2)
  @test vertex_path(ng2, (1, 1), (1, 2, 1)) == [(1, 1), (1,), (1, 2), (1, 2, 1)]
  @test edge_path(ng2, (1, 1), (1, 2, 1)) ==
    [net2((1, 1), (1,)), net2((1,), (1, 2)), net2((1, 2), (1, 2, 1))]
  @test vertex_path(ng2, (1, 1, 2), (1, 1, 1)) == [(1, 1, 2), (1, 1), (1, 1, 1)]
  @test edge_path(ng2, (1, 1, 2), (1, 1, 1)) ==
    [net2((1, 1, 2), (1, 1)), net2((1, 1), (1, 1, 1))]
end
