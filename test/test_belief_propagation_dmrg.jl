@eval module $(gensym())

using NamedGraphs.NamedGraphGenerators: named_grid
using ITensors: ITensors, Algorithm
using ITensors: siteinds
using ITensorNetworks: alternating_update, random_tensornetwork, ttn
using ITensorNetworks.ModelHamiltonians: heisenberg
using ITensorNetworks: ITensorNetwork

using Random: Random

using Test: @test, @testset

@testset "belief_propagation dmrg" begin
  ITensors.disable_warn_order()

  g = named_grid((3, 1))
  s = siteinds("S=1/2", g)
  χ = 2
  Random.seed!(1234)
  ψ = random_tensornetwork(s; link_space=χ)
  A = ITensorNetwork(ttn(heisenberg(g), s))
  
  @show alternating_update(Algorithm("bp_onesite"), A, ψ)
  
end
end
