import NetworkSolvers as ns
using Test: @test, @testset

using ITensors
import ITensorNetworks as itn
import Graphs as gr
import NamedGraphs as ng
import ITensorMPS as itm

include("utilities/simple_ed_methods.jl")
include("utilities/tree_graphs.jl")

@testset "Tree DMRG" begin
  outputlevel = 1

  g = build_tree(; nbranch=3, nbranch_sites=3)

  sites = itn.siteinds("S=1/2", g)

  # Make Heisenberg model Hamiltonian
  h = itm.OpSum()
  for edge in gr.edges(sites)
    i, j = gr.src(edge), gr.dst(edge)
    h += "Sz", i, "Sz", j
    h += 1/2, "S+", i, "S-", j
    h += 1/2, "S-", i, "S+", j
  end
  H = itn.ttn(h, sites)

  # Make initial product state
  state = Dict{Tuple{Int,Int},String}()
  for (j, v) in enumerate(gr.vertices(sites))
    state[v] = iseven(j) ? "Up" : "Dn"
  end
  psi0 = itn.ttn(state, sites)

  (outputlevel >= 1) && println("Computing exact ground state")
  Ex, psix = ed_ground_state(H, psi0)
  (outputlevel >= 1) && println("Ex = ", Ex)

  cutoff = 1E-5
  maxdim = 40
  nsweeps = 5

  #
  # Test 2-site DMRG without subspace expansion
  #
  nsites = 2
  trunc = (; cutoff, maxdim)
  inserter_kwargs = (; trunc)
  E, psi = ns.dmrg(H, psi0; inserter_kwargs, nsites, nsweeps, outputlevel)
  (outputlevel >= 1) && println("2-site DMRG energy = ", E)
  @test abs(E-Ex) < 1E-5

  #
  # Test 1-site DMRG with subspace expansion
  #
  nsites = 1
  nsweeps = 5
  trunc = (; cutoff, maxdim)
  extracter_kwargs = (; trunc, subspace_algorithm="densitymatrix")
  inserter_kwargs = (; trunc)
  cutoff = 1E-10
  maxdim = 200
  E, psi = ns.dmrg(H, psi0; extracter_kwargs, inserter_kwargs, nsites, nsweeps, outputlevel)
  (outputlevel >= 1) && println("1-site+subspace DMRG energy = ", E)
  @test abs(E-Ex) < 1E-5
end
