@eval module $(gensym())
using DataGraphs: vertex_data
using Graphs: vertices
using ITensorNetworks: ttn, contract, ortho_region, siteinds
using ITensors: @disable_warn_order, random_itensor
using LinearAlgebra: norm
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using Random: shuffle
using StableRNGs: StableRNG
using Test: @test, @testset
@testset "TTN Basics" begin
  # random comb tree
  rng = StableRNG(1234)
  tooth_lengths = rand(rng, 1:3, rand(rng, 2:4))
  c = named_comb_tree(tooth_lengths)
  # specify random site dimension on every site
  dmap = v -> rand(rng, 1:3)
  is = siteinds(dmap, c)
  # specify random linear vertex ordering of graph vertices
  vertex_order = shuffle(rng, collect(vertices(c)))

  @testset "Construct TTN from ITensor or Array" begin
    cutoff = 1e-10
    # create random ITensor with these indices
    rng = StableRNG(1234)
    S = random_itensor(rng, vertex_data(is)...)
    # dense TTN constructor from IndsNetwork
    @disable_warn_order s1 = ttn(S, is; cutoff)
    root_vertex = only(ortho_region(s1))
    @disable_warn_order begin
      S1 = contract(s1, root_vertex)
    end
    @test norm(S - S1) < 1e2 * cutoff
  end

  @testset "Ortho" begin
    # TODO
  end
end
end
