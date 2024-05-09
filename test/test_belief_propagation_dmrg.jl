@eval module $(gensym())

using NamedGraphs: nv
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree, named_binary_tree
using ITensors: ITensors, Algorithm
using ITensors: dmrg, siteinds
using ITensorNetworks: alternating_update, bp_inserter, bp_extracter, bp_eigsolve_updater, random_tensornetwork, ttn, inner, maxlinkdim
using ITensorNetworks.ModelHamiltonians: heisenberg
using ITensorNetworks: ITensorNetwork

using Random: Random

using Test: @test, @testset

@testset "belief_propagation dmrg" begin
  ITensors.disable_warn_order()

  g = named_binary_tree(4)
  s = siteinds("S=1/2", g)
  χ, χmax = 1, 10
  Random.seed!(1234)
  ψ = random_tensornetwork(s; link_space=χ)
  A = ITensorNetwork(ttn(heisenberg(g), s))
  inserter_kwargs = (; maxdim = χmax)
  updater_kwargs = (; tol = 1e-14, krylovdim = 1, maxiter = 1, verbosity = 0, eager = false)
  nsites, nsweeps = 2, 10
  
  @time e_bp, ψ_bp = dmrg(copy(A), copy(ψ); nsweeps, nsites, updater_kwargs, inserter_kwargs)
  @time e_ttn, ψ_ttn = dmrg(ttn(A), ttn(ψ); nsweeps, nsites, inserter_kwargs, updater_kwargs)
  
  @show e_ttn, e_bp
end
end
