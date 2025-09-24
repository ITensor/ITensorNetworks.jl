using Test: @test, @testset
using ITensors
using ITensorNetworks: ITensorNetworks, applyexp, dmrg, maxlinkdim, siteinds, time_evolve, ttn
using Graphs: add_vertex!, add_edge!, vertices
using NamedGraphs: NamedGraph
using NamedGraphs.NamedGraphGenerators: named_path_graph
using ITensorMPS: OpSum
using TensorOperations: TensorOperations #For contraction order finding

function chain_plus_ancilla(; nchain)
  g = NamedGraph()
  for j in 1:nchain
    add_vertex!(g, j)
  end
  for j in 1:(nchain-1)
    add_edge!(g, j => j + 1)
  end
  # Add ancilla vertex near middle of chain
  add_vertex!(g, 0)
  add_edge!(g, 0 => nchain ÷ 2)
  return g
end

@testset "Test Tree Time Evolution" begin
  outputlevel = 0

  N = 10
  g = chain_plus_ancilla(; nchain=N)

  sites = siteinds("S=1/2", g)

  # Make Heisenberg model Hamiltonian
  h = OpSum()
  for j in 1:(N-1)
    h += "Sz", j, "Sz", j + 1
    h += 1 / 2, "S+", j, "S-", j + 1
    h += 1 / 2, "S-", j, "S+", j + 1
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
  E, gs_psi = dmrg(H, psi0; insert_kwargs=(; trunc), nsites, nsweeps, outputlevel)
  (outputlevel >= 1) && println("2-site DMRG energy = ", E)

  insert_kwargs = (; trunc)
  nsites = 1
  tmax = 0.10
  time_range = 0.0:0.02:tmax
  psi1_t = time_evolve(H, time_range, gs_psi; insert_kwargs, nsites, outputlevel)
  (outputlevel >= 1) && println("Done with $nsites-site TDVP")

  @test norm(psi1_t) > 0.999

  nsites = 2
  psi2_t = time_evolve(H, time_range, gs_psi; insert_kwargs, nsites, outputlevel)
  (outputlevel >= 1) && println("Done with $nsites-site TDVP")
  @test norm(psi2_t) > 0.999

  @test abs(inner(psi1_t, gs_psi)) > 0.99
  @test abs(inner(psi1_t, psi2_t)) > 0.99

  # Test that accumulated phase angle is E*tmax
  z = inner(psi1_t, gs_psi)
  @test atan(imag(z) / real(z)) ≈ E * tmax atol = 1E-4
end

@testset "Applyexp Time Point Handling" begin
  N = 10
  g = named_path_graph(N)
  sites = siteinds("S=1/2", g)

  # Make Heisenberg model Hamiltonian
  h = OpSum()
  for j in 1:(N-1)
    h += "Sz", j, "Sz", j + 1
    h += 1 / 2, "S+", j, "S-", j + 1
    h += 1 / 2, "S-", j, "S+", j + 1
  end
  H = ttn(h, sites)

  # Initial product state
  state = Dict{Int,String}()
  for (j, v) in enumerate(vertices(sites))
    state[v] = iseven(j) ? "Up" : "Dn"
  end
  psi0 = ttn(state, sites)

  nsites = 2
  trunc = (; cutoff=1E-8, maxdim=100)
  insert_kwargs = (; trunc)

  # Test that all time points are reached and reported correctly
  time_points = [0.0, 0.1, 0.25, 0.32, 0.4]
  times = Real[]
  function collect_times(problem; kws...)
    push!(times, ITensorNetworks.current_time(problem))
  end
  time_evolve(H, time_points, psi0; insert_kwargs, nsites, sweep_callback=collect_times, outputlevel=1)
  @test times ≈ time_points atol = 10 * eps(Float64)

  # Test that all exponents are reached and reported correctly
  exponent_points = [-0.0, -0.1, -0.25, -0.32, -0.4]
  exponents = Real[]
  function collect_exponents(problem; kws...)
    push!(exponents, ITensorNetworks.current_exponent(problem))
  end
  applyexp(H, exponent_points, psi0; insert_kwargs, nsites, sweep_callback=collect_exponents, outputlevel=1)
  @test exponents ≈ exponent_points atol = 10 * eps(Float64)
end
