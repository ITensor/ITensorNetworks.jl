using ITensors
using ITensorNetworks
using ITensorUnicodePlots
using Random
using Test

@testset "ITensorNetworks.jl" begin
  Random.seed!(1234)
  g = chain_lattice_graph(4)
  s = siteinds("S=1/2", g)

  @test s isa IndsNetwork
  @test nv(s) == 4
  @test ne(s) == 3
  @test neighbors(s, 2) == [1, 3]

  tn = ITensorNetwork(s; link_space=2)

  @test nv(tn) == 4
  @test ne(tn) == 3
  @test tn isa ITensorNetwork
  @test neighbors(tn, 2) == [1, 3]
  @test tn[1] isa ITensor
  @test order(tn[1]) == 2
  @test tn[2] isa ITensor
  @test order(tn[2]) == 3
  @test tn[1:2] isa ITensorNetwork

  randn!.(vertex_data(tn))
  tn′ = sim(dag(tn); sites=[])

  @test tn′ isa ITensorNetwork

  ## inner_tn = tn ⊗ tn′

  ## @test inner_tn isa ITensorNetwork

  ## sequence = optimal_contraction_sequence(inner_tn)

  ## @test sequence isa Vector

  ## inner_res = contract(inner_tn; sequence)[]

  ## @test inner_res isa Float64
end
