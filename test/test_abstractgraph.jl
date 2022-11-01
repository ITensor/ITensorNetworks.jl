using Test
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
