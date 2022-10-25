using ITensors
using Test
using ITensorNetworks
using ITensorNetworks.ApproximateTNContraction: itensor_network, neighbors

@testset "itensor_network.jl" begin
  @testset "itensor_network from dims" begin
    d = (3, 3)
    tn = itensor_network(d...; linkdims=3)
    @test tn isa Matrix{ITensor}
    @test size(tn) == (3, 3)
    @test all(ITensors.isemptystorage, tn)
    @test hascommoninds(tn[1, 1], tn[1, 2])
    @test hascommoninds(tn[1, 1], tn[2, 1])
    @test !hascommoninds(tn[1, 1], tn[2, 2])
    @test hascommoninds(tn[1, 1], tn[3, 1])
    @test hascommoninds(tn[1, 1], tn[1, 3])
    @test isempty(uniqueinds(tn[2, 2], tn[2, 1], tn[2, 3], tn[1, 2], tn[3, 2]))
    @test issetequal(
      neighbors(tn, CartesianIndex(1, 2)),
      [
        CartesianIndex(1, 1),
        CartesianIndex(2, 2),
        CartesianIndex(3, 2),
        CartesianIndex(1, 3),
      ],
    )
  end
  @testset "itensor_network from siteinds" begin
    d = (3, 3)
    s = siteinds("S=1/2", d...)
    tn = itensor_network(s; linkdims=3)
    @test tn isa Matrix{ITensor}
    @test size(tn) == (3, 3)
    @test all(ITensors.isemptystorage, tn)
    @test hascommoninds(tn[1, 1], tn[1, 2])
    @test hascommoninds(tn[1, 1], tn[2, 1])
    @test !hascommoninds(tn[1, 1], tn[2, 2])
    @test hascommoninds(tn[1, 1], tn[3, 1])
    @test hascommoninds(tn[1, 1], tn[1, 3])
    @test uniqueinds(tn[2, 2], tn[2, 1], tn[2, 3], tn[1, 2], tn[3, 2]) == [s[2, 2]]
    @test issetequal(
      neighbors(tn, CartesianIndex(1, 2)),
      [
        CartesianIndex(1, 1),
        CartesianIndex(2, 2),
        CartesianIndex(3, 2),
        CartesianIndex(1, 3),
      ],
    )
  end
end
