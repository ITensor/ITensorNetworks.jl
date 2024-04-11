@eval module $(gensym())
using DataGraphs: vertex_data
using Graphs: vertices
using ITensorNetworks: ttn, contract, ortho_center, siteinds
using ITensors: @disable_warn_order, randomITensor
using LinearAlgebra: norm
using NamedGraphs: named_comb_tree
using Random: shuffle
using Test: @test, @testset

@testset "TTN Basics" begin

  # random comb tree
  tooth_lengths = rand(1:3, rand(2:4))
  c = named_comb_tree(tooth_lengths)
  # specify random site dimension on every site
  dmap = v -> rand(1:3)
  is = siteinds(dmap, c)
  # specify random linear vertex ordering of graph vertices
  vertex_order = shuffle(vertices(c))

  @testset "Construct TTN from ITensor or Array" begin
    cutoff = 1e-10
    # create random ITensor with these indices
    S = randomITensor(vertex_data(is)...)
    # dense TTN constructor from IndsNetwork
    @disable_warn_order s1 = ttn(S, is; cutoff)
    root_vertex = only(ortho_center(s1))
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
