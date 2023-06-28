using Compat
using ITensors
using Metis
using ITensorNetworks
using Random
using SplitApplyCombine
using ProfileView

using ITensorNetworks:
  message_tensors,
  nested_graph_leaf_vertices,
  belief_propagation_iteration,
  belief_propagation,
  find_subgraph,
  vidal_gauge,
  symmetric_gauge,
  vidal_itn_canonicalness,
  vidal_to_symmetric_gauge,
  initialize_bond_tensors,
  vidal_itn_isometries,
  unflattened_norm_network

using NamedGraphs
using NamedGraphs: add_edges!, rem_vertex!, hexagonal_lattice_graph
using Graphs

"""Eager Gauging"""
function eager_gauging(ψ::ITensorNetwork, bond_tensors::DataGraph, mts::DataGraph)
  isometries = vidal_itn_isometries(ψ, bond_tensors)

  ψ = copy(ψ)
  mts = copy(mts)
  for e in edges(ψ)
    s1, s2 = find_subgraph((src(e), 1), mts), find_subgraph((dst(e), 1), mts)
    normalize!(isometries[e])
    normalize!(isometries[reverse(e)])
    mts[s1 => s2], mts[s2 => s1] = ITensorNetwork(isometries[e]),
    ITensorNetwork(isometries[reverse(e)])
  end

  ψ, bond_tensors = vidal_gauge(ψ, mts, bond_tensors)

  return ψ, bond_tensors, mts
end

"""Bring an ITN into the Vidal gauge, various methods possible. Result is timed"""
function benchmark_state_gauging(
  ψ::ITensorNetwork; mode="BeliefPropagation", no_iterations=50
)
  s = siteinds(ψ)

  C = zeros((no_iterations))
  times_iters = zeros((no_iterations))
  times_gauging = zeros((no_iterations))

  ψψ = unflattened_norm_network(ψ)
  ψinit = copy(ψ)
  vertex_groups = nested_graph_leaf_vertices(partition(ψψ, group(v -> v[1], vertices(ψψ))))
  mts = message_tensors(partition(ψψ; subgraph_vertices=vertex_groups))
  bond_tensors = initialize_bond_tensors(ψ)
  for e in edges(mts)
    mts[e] = ITensorNetwork(dense(delta(inds(ITensors.contract(ITensor(mts[e]))))))
  end

  for i in 1:no_iterations
    println("On Iteration " * string(i))

    if mode == "BeliefPropagation"
      times_iters[i] = @elapsed mts, _ = belief_propagation_iteration(
        ψψ, mts; contract_kwargs=(; alg="exact")
      )
      times_gauging[i] = @elapsed ψ, bond_tensors = vidal_gauge(ψinit, mts)
    elseif mode == "Eager"
      times_iters[i] = @elapsed ψ, bond_tensors, mts = eager_gauging(ψ, bond_tensors, mts)
    else
      times_iters[i] = @elapsed begin
        for e in edges(ψ)
          ψ, bond_tensors = apply(e, ψ, bond_tensors; normalize=true, cutoff=1e-16)
        end
      end
    end

    C[i] = vidal_itn_canonicalness(ψ, bond_tensors)
  end

  simulation_times = cumsum(times_iters) + times_gauging

  return simulation_times, C
end

L, χ = 10, 10
g = named_grid((L, L))
s = siteinds("S=1/2", g)
ψ = randomITensorNetwork(s; link_space=χ)

BPG_simulation_times, BPG_Cs = benchmark_state_gauging(ψ; no_iterations=40)
Eager_simulation_times, Eager_Cs = benchmark_state_gauging(
  ψ; mode="Eager", no_iterations=40
)
SU_simulation_times, SU_Cs = benchmark_state_gauging(ψ; mode="SU", no_iterations=40)

epsilon = 1e-8
println(
  "Time for BPG to reach C < epsilon was " *
  string(BPG_simulation_times[findfirst(x -> x < 0, BPG_Cs .- epsilon)]) *
  " seconds",
)
println(
  "Time for Eager to reach C < epsilon was " *
  string(Eager_simulation_times[findfirst(x -> x < 0, Eager_Cs .- epsilon)]) *
  " seconds",
)
println(
  "Time for SU to reach C < epsilon was " *
  string(SU_simulation_times[findfirst(x -> x < 0, SU_Cs .- epsilon)]) *
  " seconds",
)
