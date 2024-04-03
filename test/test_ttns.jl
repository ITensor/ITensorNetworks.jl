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
    sites_s = [only(is[v]) for v in vertex_order]
    # create random ITensor with these indices
    S = randomITensor(vertex_data(is)...)
    # dense TTN constructor from IndsNetwork
    @disable_warn_order s1 = ttn(S, is; cutoff)
    # dense TTN constructor from Vector{Index} and NamedDimGraph
    @disable_warn_order s2 = ttn(S, sites_s, c; vertex_order, cutoff)
    # convert to array with proper index order
    @disable_warn_order AS = Array(S, sites_s...)
    # dense array constructor from IndsNetwork
    @disable_warn_order s3 = ttn(AS, is; vertex_order, cutoff)
    # dense array constructor from Vector{Index} and NamedDimGraph
    @disable_warn_order s4 = ttn(AS, sites_s, c; vertex_order, cutoff)
    # see if this actually worked
    root_vertex = only(ortho_center(s1))
    @disable_warn_order begin
      S1 = contract(s1, root_vertex)
      S2 = contract(s2, root_vertex)
      S3 = contract(s3, root_vertex)
      S4 = contract(s4, root_vertex)
    end
    @test norm(S - S1) < 1e2 * cutoff
    @test norm(S - S2) < 1e2 * cutoff
    @test norm(S - S3) < 1e2 * cutoff
    @test norm(S - S4) < 1e2 * cutoff
  end

  @testset "Ortho" begin
    # TODO
  end
end
end
