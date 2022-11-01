using ITensors
using ITensorNetworks
using ITensorUnicodePlots
using Random
using Test

include("test_tebd.jl")

@testset "ITensorNetworks.jl" begin
  @testset "Basics" begin
    Random.seed!(1234)
    g = named_grid((4,))
    s = siteinds("S=1/2", g)

    @test s isa IndsNetwork
    @test nv(s) == 4
    @test ne(s) == 3
    @test neighbors(s, 2) == [(1,), (3,)]

    tn = ITensorNetwork(s; link_space=2)

    @test nv(tn) == 4
    @test ne(tn) == 3
    @test tn isa ITensorNetwork
    @test neighbors(tn, 2) == [(1,), (3,)]
    @test tn[1] isa ITensor
    @test order(tn[1]) == 2
    @test tn[2] isa ITensor
    @test order(tn[2]) == 3
    @test tn[1:2] isa ITensorNetwork

    randn!.(vertex_data(tn))
    tn′ = sim(dag(tn); sites=[])

    @test tn′ isa ITensorNetwork
  end

  @testset "Contract edge (regression test for issue #5)" begin
    dims = (2, 2)
    g = named_grid(dims)
    s = siteinds("S=1/2", g)
    ψ = ITensorNetwork(s, v -> "↑")
    tn = inner(ψ, sim(dag(ψ), sites=[]))
    tn_2 = contract(tn,  (2, 1, 2) => (1, 1, 2))
    @test !has_vertex(tn_2, (2, 1, 2))
    @test tn_2[1, 1, 2] ≈ tn[2, 1, 2] * tn[1, 1, 2]
  end

  @testset "Remove edge (regression test for issue #5)" begin
    dims = (2, 2)
    g = named_grid(dims)
    s = siteinds("S=1/2", g)
    ψ = ITensorNetwork(s, v -> "↑")
    rem_vertex!(ψ, (1, 2))
    tn = inner(ψ, sim(dag(ψ), sites=[]))
    @test !has_vertex(tn, (1, 1, 2))
    @test !has_vertex(tn, (2, 1, 2))
    @test has_vertex(tn, (1, 1, 1))
    @test has_vertex(tn, (2, 1, 1))
    @test has_vertex(tn, (1, 2, 1))
    @test has_vertex(tn, (2, 2, 1))
    @test has_vertex(tn, (1, 2, 2))
    @test has_vertex(tn, (2, 2, 2))
  end

  ## inner_tn = tn ⊗ tn′

  ## @test inner_tn isa ITensorNetwork

  ## sequence = optimal_contraction_sequence(inner_tn)

  ## @test sequence isa Vector

  ## inner_res = contract(inner_tn; sequence)[]

  ## @test inner_res isa Float64
end
