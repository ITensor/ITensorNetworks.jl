@eval module $(gensym())
using Graphs: vertices
using ITensorNetworks: ttn, contract, ortho_region, siteinds, union_all_inds
using ITensors: @disable_warn_order, prime, random_itensor
using LinearAlgebra: norm
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using Random: shuffle
using StableRNGs: StableRNG
using Test: @test, @testset
@testset "TTN operator Basics" begin
  # random comb tree
  rng = StableRNG(1234)
  tooth_lengths = rand(rng, 1:3, rand(rng, 2:4))
  c = named_comb_tree(tooth_lengths)
  # specify random site dimension on every site
  dmap = v -> rand(rng, 1:3)
  is = siteinds(dmap, c)
  # operator site inds
  is_isp = union_all_inds(is, prime(is; links=[]))
  # specify random linear vertex ordering of graph vertices
  vertex_order = shuffle(rng, collect(vertices(c)))
  @testset "Construct TTN operator from ITensor or Array" begin
    cutoff = 1e-10
    sites_o = [is_isp[v] for v in vertex_order]
    # create random ITensor with these indices
    rng = StableRNG(1234)
    O = random_itensor(rng, sites_o...)
    # dense TTN constructor from IndsNetwork
    @disable_warn_order o1 = ttn(O, is_isp; cutoff)
    root_vertex = only(ortho_region(o1))
    @disable_warn_order begin
      O1 = contract(o1, root_vertex)
    end
    @test norm(O - O1) < 1e2 * cutoff
  end
  @testset "Ortho" begin
    # TODO
  end
end
end
