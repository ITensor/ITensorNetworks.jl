using ITensors
using ITensorNetworks
using Random
using Test

@testset "ITensorNetwork Basics" begin
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
  inner_tn = tn ⊗ tn′
  @test inner_tn isa ITensorNetwork
  sequence = contraction_sequence(inner_tn)
  @test sequence isa Vector
  inner_res = contract(inner_tn; sequence)[]
  @test inner_res isa Float64

  @testset "Contract edge (regression test for issue #5)" begin
    dims = (2, 2)
    g = named_grid(dims)
    s = siteinds("S=1/2", g)
    ψ = ITensorNetwork(s, v -> "↑")
    tn = inner_network(ψ, sim(dag(ψ); sites=[]))
    tn_2 = contract(tn, (2, 1, 2) => (1, 1, 2))
    @test !has_vertex(tn_2, (2, 1, 2))
    @test tn_2[1, 1, 2] ≈ tn[2, 1, 2] * tn[1, 1, 2]
  end

  @testset "Remove edge (regression test for issue #5)" begin
    dims = (2, 2)
    g = named_grid(dims)
    s = siteinds("S=1/2", g)
    ψ = ITensorNetwork(s, v -> "↑")
    rem_vertex!(ψ, (1, 2))
    tn = inner_network(ψ, sim(dag(ψ); sites=[]))
    @test !has_vertex(tn, (1, 1, 2))
    @test !has_vertex(tn, (2, 1, 2))
    @test has_vertex(tn, (1, 1, 1))
    @test has_vertex(tn, (2, 1, 1))
    @test has_vertex(tn, (1, 2, 1))
    @test has_vertex(tn, (2, 2, 1))
    @test has_vertex(tn, (1, 2, 2))
    @test has_vertex(tn, (2, 2, 2))
  end

  @testset "Index access" begin
    dims = (2, 2)
    g = named_grid(dims)
    s = siteinds("S=1/2", g)
    ψ = ITensorNetwork(s; link_space=2)

    nt = ITensorNetworks.neighbor_itensors(ψ, 1, 1)
    @test length(nt) == 2
    @test all(map(hascommoninds(ψ[1, 1]), nt))

    @test all(map(t -> isempty(commoninds(inds(t), uniqueinds(ψ, 1, 1))), nt))

    e = (1, 1) => (2, 1)
    uie = uniqueinds(ψ, e)
    @test isempty(commoninds(uie, inds(ψ[2, 1])))
    @test issetequal(uie, union(commoninds(ψ[1, 1], ψ[1, 2]), uniqueinds(ψ, 1, 1)))

    @test siteinds(all, ψ, 1, 1) == s[1, 1]
    @test siteinds(only, ψ, 1, 1) == only(s[1, 1])

    cie = commoninds(ψ, e)
    @test hasinds(ψ[1, 1], cie) && hasinds(ψ[2, 1], cie)
    @test isempty(commoninds(uie, cie))

    @test linkinds(all, ψ, e) == commoninds(ψ[1, 1], ψ[2, 1])
    @test linkinds(only, ψ, e) == only(commoninds(ψ[1, 1], ψ[2, 1]))
  end

  @testset "ElType in constructors" begin
    # TODO
  end

  @testset "Construction from state (function)" begin
    # TODO
  end
end
