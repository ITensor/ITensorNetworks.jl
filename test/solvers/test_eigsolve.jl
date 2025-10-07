using Test: @test, @testset
using ITensors
using ITensorNetworks: siteinds, ttn, dmrg
using Graphs: dst, edges, src, vertices
using ITensorMPS: OpSum
using TensorOperations: TensorOperations #For contraction order finding

include("utilities/simple_ed_methods.jl")
include("utilities/tree_graphs.jl")

@testset "Tree DMRG" begin
  outputlevel = 0

  g = build_tree(; nbranch=3, nbranch_sites=3)

  sites = siteinds("S=1/2", g)

  # Make Heisenberg model Hamiltonian
  h = OpSum()
  for edge in edges(sites)
    i, j = src(edge), dst(edge)
    h += "Sz", i, "Sz", j
    h += 1 / 2, "S+", i, "S-", j
    h += 1 / 2, "S-", i, "S+", j
  end
  H = ttn(h, sites)

  # Make initial product state
  state = Dict{Tuple{Int,Int},String}()
  for (j, v) in enumerate(vertices(sites))
    state[v] = iseven(j) ? "Up" : "Dn"
  end
  psi0 = ttn(state, sites)

  (outputlevel >= 1) && println("Computing exact ground state")
  Ex, psix = ed_ground_state(H, psi0)
  (outputlevel >= 1) && println("Ex = ", Ex)

  cutoff = 1E-5
  maxdim = 40

  factorize_kwargs = (; cutoff, maxdim)

  nsweeps = 5

  #
  # Test 2-site DMRG without subspace expansion
  #
  nsites = 2
  E, psi = dmrg(H, psi0; factorize_kwargs, nsites, nsweeps, outputlevel)
  (outputlevel >= 1) && println("2-site DMRG energy = ", E)
  @test E ≈ Ex atol = 1E-5

  #
  # Test 1-site DMRG with subspace expansion
  #
  nsites = 1
  nsweeps = 5
  extract!_kwargs = (; subspace_algorithm="densitymatrix")
  E, psi = dmrg(H, psi0; extract!_kwargs, factorize_kwargs, nsites, nsweeps, outputlevel)
  (outputlevel >= 1) && println("1-site+subspace DMRG energy = ", E)
  @test E ≈ Ex atol = 1E-5
end
