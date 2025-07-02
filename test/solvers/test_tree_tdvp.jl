using Test: @test, @testset
using ITensors
using TensorOperations # Needed to use contraction order finding
import ITensorNetworks: dmrg, maxlinkdim, siteinds, time_evolve, ttn
import Graphs: add_vertex!, add_edge!, vertices
import NamedGraphs: NamedGraph
import ITensorMPS: OpSum

function chain_plus_ancilla(; nchain)
  g = NamedGraph()
  for j in 1:nchain
    add_vertex!(g, j)
  end
  for j in 1:(nchain - 1)
    add_edge!(g, j=>j+1)
  end
  # Add ancilla vertex near middle of chain
  add_vertex!(g, 0)
  add_edge!(g, 0=>nchain÷2)
  return g
end

@testset "Tree TDVP on chain plus ancilla" begin
  outputlevel = 1

  N = 10
  g = chain_plus_ancilla(; nchain=N)

  sites = siteinds("S=1/2", g)

  # Make Heisenberg model Hamiltonian
  h = OpSum()
  for j in 1:(N - 1)
    h += "Sz", j, "Sz", j+1
    h += 1/2, "S+", j, "S-", j+1
    h += 1/2, "S-", j, "S+", j+1
  end
  H = ttn(h, sites)

  # Make initial product state
  state = Dict{Int,String}()
  for (j, v) in enumerate(vertices(sites))
    state[v] = iseven(j) ? "Up" : "Dn"
  end
  psi0 = ttn(state, sites)

  cutoff = 1E-10
  maxdim = 100
  nsweeps = 5

  nsites = 2
  trunc = (; cutoff, maxdim)
  E, gs_psi = dmrg(H, psi0; inserter_kwargs=(; trunc), nsites, nsweeps, outputlevel)
  (outputlevel >= 1) && println("2-site DMRG energy = ", E)

  inserter_kwargs=(; trunc)
  nsites = 1
  tmax = 0.10
  time_range = 0.0:0.02:tmax
  psi1_t = time_evolve(H, gs_psi, time_range; inserter_kwargs, nsites, outputlevel)
  (outputlevel >= 1) && println("Done with $nsites-site TDVP")

  @test norm(psi1_t) > 0.999

  nsites = 2
  psi2_t = time_evolve(H, gs_psi, time_range; inserter_kwargs, nsites, outputlevel)
  (outputlevel >= 1) && println("Done with $nsites-site TDVP")
  @test norm(psi2_t) > 0.999

  @test abs(inner(psi1_t, gs_psi)) > 0.99
  @test abs(inner(psi1_t, psi2_t)) > 0.99

  # Test that accumulated phase angle is E*tmax
  z = inner(psi1_t, gs_psi)
  @test abs(atan(imag(z)/real(z)) - E*tmax) < 1E-4
end
