using ITensors
using ITensorNetworks
using ITensorUnicodePlots
using Random
using Test

@testset "ITensorNetworks.jl" begin
  Random.seed!(1234)
  g = chain_lattice_graph(4)
  s = siteinds("S=1/2", g)
  tn = ITensorNetwork(s; link_space=2)

  @test tn isa ITensorNetwork
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
