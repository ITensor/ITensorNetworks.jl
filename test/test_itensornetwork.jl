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
  @test neighbors(s, (2,)) == [(1,), (3,)]

  tn = ITensorNetwork(s; link_space=2)

  @test nv(tn) == 4
  @test ne(tn) == 3
  @test tn isa ITensorNetwork
  @test neighbors(tn, (2,)) == [(1,), (3,)]

  # TODO: How to support this syntax?
  @test_broken tn[1] isa ITensor
  @test tn[(1,)] isa ITensor

  # TODO: How to support this syntax?
  @test_broken order(tn[1]) == 2
  @test order(tn[(1,)]) == 2

  # TODO: How to support this syntax?
  @test_broken tn[2] isa ITensor
  @test tn[(2,)] isa ITensor

  # TODO: How to support this syntax?
  @test_broken order(tn[2]) == 3
  @test order(tn[(2,)]) == 3

  # XXX: Slicing syntax is no long supported, use `induced_subgraph`.
  @test_broken tn[1:2] isa ITensorNetwork
  # TODO: Support this syntax, maybe rename `subgraph`.
  @test_broken induced_subgraph(tn, [(1,), (2,)]) isa ITensorNetwork

  randn!.(vertex_data(tn))
  tn′ = sim(dag(tn); sites=[])

  @test tn′ isa ITensorNetwork
  inner_tn = tn ⊗ tn′
  @test inner_tn isa ITensorNetwork
  sequence = contraction_sequence(inner_tn)
  @test sequence isa Vector
  inner_res = contract(inner_tn; sequence)[]
  @test inner_res isa Float64
end

@testset "Contract edge (regression test for issue #5)" begin
  dims = (2, 2)
  g = named_grid(dims)
  s = siteinds("S=1/2", g)
  ψ = ITensorNetwork(s, v -> "↑")
  tn = inner_network(ψ, ψ)
  tn_2 = contract(tn, ((1, 2), 2) => ((1, 2), 1))
  @test !has_vertex(tn_2, ((1, 2), 2))
  @test tn_2[((1, 2), 1)] ≈ tn[((1, 2), 2)] * tn[((1, 2), 1)]
end

@testset "Remove edge (regression test for issue #5)" begin
  dims = (2, 2)
  g = named_grid(dims)
  s = siteinds("S=1/2", g)
  ψ = ITensorNetwork(s, v -> "↑")
  rem_vertex!(ψ, (1, 2))
  tn = inner_network(ψ, ψ)
  @test !has_vertex(tn, ((1, 2), 1))
  @test !has_vertex(tn, ((1, 2), 2))
  @test has_vertex(tn, ((1, 1), 1))
  @test has_vertex(tn, ((1, 1), 2))
  @test has_vertex(tn, ((2, 1), 1))
  @test has_vertex(tn, ((2, 1), 2))
  @test has_vertex(tn, ((2, 2), 1))
  @test has_vertex(tn, ((2, 2), 2))
end

@testset "Custom element type" for eltype in (Float32, Float64, ComplexF32, ComplexF64),
  link_space in (nothing, 3),
  g in (
    grid((4,)),
    named_grid((3, 3)),
    siteinds("S=1/2", grid((4,))),
    siteinds("S=1/2", named_grid((3, 3))),
  )
  ψ = ITensorNetwork(g; link_space) do v, inds...
    return itensor(randn(eltype, dims(inds)...), inds...)
  end
  @test Base.eltype(ψ[first(vertices(ψ))]) == eltype
  ψ = ITensorNetwork(g; link_space) do v, inds...
    return itensor(randn(dims(inds)...), inds...)
  end
  @test Base.eltype(ψ[first(vertices(ψ))]) == Float64
  ψ = randomITensorNetwork(eltype, g; link_space)
  @test Base.eltype(ψ[first(vertices(ψ))]) == eltype
  ψ = randomITensorNetwork(g; link_space)
  @test Base.eltype(ψ[first(vertices(ψ))]) == Float64
  ψ = ITensorNetwork(eltype, undef, g; link_space)
  @test Base.eltype(ψ[first(vertices(ψ))]) == eltype
  ψ = ITensorNetwork(undef, g)
  @test Base.eltype(ψ[first(vertices(ψ))]) == Float64
end
