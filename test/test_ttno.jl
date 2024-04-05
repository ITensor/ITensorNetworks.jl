@eval module $(gensym())
using Graphs: vertices
using ITensorNetworks: ttn, contract, ortho_center, siteinds, union_all_inds
using ITensors: @disable_warn_order, prime, randomITensor
using LinearAlgebra: norm
using NamedGraphs: named_comb_tree
using Random: shuffle
using Test: @test, @testset

@testset "TTN operator Basics" begin

  # random comb tree
  tooth_lengths = rand(1:3, rand(2:4))
  c = named_comb_tree(tooth_lengths)
  # specify random site dimension on every site
  dmap = v -> rand(1:3)
  is = siteinds(dmap, c)
  # operator site inds
  is_isp = union_all_inds(is, prime(is; links=[]))
  # specify random linear vertex ordering of graph vertices
  vertex_order = shuffle(vertices(c))

  @testset "Construct TTN operator from ITensor or Array" begin
    cutoff = 1e-10
    sites_o = [is_isp[v] for v in vertex_order]
    # create random ITensor with these indices
    O = randomITensor(sites_o...)
    # dense TTN constructor from IndsNetwork
    @disable_warn_order o1 = ttn(O, is_isp; cutoff)
    root_vertex = only(ortho_center(o1))
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
