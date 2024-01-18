using Compat
using ITensors
using Metis
using ITensorNetworks
using Random
using SplitApplyCombine

using ITensorNetworks:
  message_tensors,
  belief_propagation,
  vidal_gauge,
  symmetric_gauge,
  vidal_itn_canonicalness,
  initialize_bond_tensors,
  vidal_itn_isometries,
  norm_network,
  edge_sequence,
  belief_propagation_iteration

using NamedGraphs
using NamedGraphs: add_edges!, rem_vertex!, hexagonal_lattice_graph
using Graphs

"""Eager Gauging"""
function eager_gauging(
  ψ::ITensorNetwork, pψψ::PartitionedGraph, bond_tensors::DataGraph, mts
)
  isometries = vidal_itn_isometries(ψ, bond_tensors)

  ψ = copy(ψ)
  mts = copy(mts)
  for e in edges(ψ)
    vsrc, vdst = src(e), dst(e)
    pe = NamedGraphs.partition_edge(pψψ, NamedEdge((vsrc, 1) => (vdst, 1)))
    normalize!(isometries[e])
    normalize!(isometries[reverse(e)])
    mts[pe], mts[PartitionEdge(reverse(NamedGraphs.parent(pe)))] = ITensorNetwork(
      isometries[e]
    ),
    ITensorNetwork(isometries[reverse(e)])
  end

  ψ, bond_tensors = vidal_gauge(ψ, pψψ, mts, bond_tensors)

  return ψ, bond_tensors, mts
end

"""Bring an ITN into the Vidal gauge, various methods possible. Result is timed"""
function benchmark_state_gauging(
  ψ::ITensorNetwork;
  mode="belief_propagation",
  no_iterations=50,
  BP_update_order::String="sequential",
)
  s = siteinds(ψ)

  C = zeros((no_iterations))
  times_iters = zeros((no_iterations))
  times_gauging = zeros((no_iterations))

  ψψ = norm_network(ψ)
  ψinit = copy(ψ)
  pψψ = PartitionedGraph(ψψ, collect(values(group(v -> v[1], vertices(ψψ)))))
  mts = message_tensors(pψψ)
  bond_tensors = initialize_bond_tensors(ψ)

  for i in 1:no_iterations
    println("On Iteration " * string(i))

    if mode == "belief_propagation"
      if BP_update_order != "parallel"
        times_iters[i] = @elapsed mts, _ = belief_propagation_iteration(
          pψψ, mts; contract_kwargs=(; alg="exact")
        )
      else
        times_iters[i] = @elapsed mts, _ = belief_propagation_iteration(
          pψψ,
          mts;
          contract_kwargs=(; alg="exact"),
          edges=[
            PartitionEdge.(e) for e in edge_sequence(partitioned_graph(pψψ); alg="parallel")
          ],
        )
      end

      times_gauging[i] = @elapsed ψ, bond_tensors = vidal_gauge(ψinit, pψψ, mts)
    elseif mode == "eager"
      times_iters[i] = @elapsed ψ, bond_tensors, mts = eager_gauging(
        ψ, pψψ, bond_tensors, mts
      )
    else
      times_iters[i] = @elapsed begin
        for e in edges(ψ)
          ψ, bond_tensors = apply(e, ψ, bond_tensors; normalize=true, cutoff=1e-16)
        end
      end
    end

    C[i] = vidal_itn_canonicalness(ψ, bond_tensors)
  end
  @show times_iters, time
  simulation_times = cumsum(times_iters) + times_gauging

  return simulation_times, C
end

L, χ = 10, 10
g = named_grid((L, L))
s = siteinds("S=1/2", g)
ψ = randomITensorNetwork(s; link_space=χ)
no_iterations = 30

BPG_simulation_times, BPG_Cs = benchmark_state_gauging(
  ψ; no_iterations, BP_update_order="parallel"
)
BPG_sequential_simulation_times, BPG_sequential_Cs = benchmark_state_gauging(
  ψ; no_iterations
)
Eager_simulation_times, Eager_Cs = benchmark_state_gauging(ψ; mode="eager", no_iterations)
SU_simulation_times, SU_Cs = benchmark_state_gauging(ψ; mode="SU", no_iterations)

epsilon = 1e-10

println(
  "Time for BPG (with parallel updates) to reach C < epsilon was " *
  string(BPG_simulation_times[findfirst(x -> x < 0, BPG_Cs .- epsilon)]) *
  " seconds. No iters was " *
  string(findfirst(x -> x < 0, BPG_Cs .- epsilon)),
)
println(
  "Time for BPG (with sequential updates) to reach C < epsilon was " *
  string(
    BPG_sequential_simulation_times[findfirst(x -> x < 0, BPG_sequential_Cs .- epsilon)]
  ) *
  " seconds. No iters was " *
  string(findfirst(x -> x < 0, BPG_sequential_Cs .- epsilon)),
)

println(
  "Time for Eager Gauging to reach C < epsilon was " *
  string(Eager_simulation_times[findfirst(x -> x < 0, Eager_Cs .- epsilon)]) *
  " seconds. No iters was " *
  string(findfirst(x -> x < 0, Eager_Cs .- epsilon)),
)
println(
  "Time for SU Gauging (with sequential updates) to reach C < epsilon was " *
  string(SU_simulation_times[findfirst(x -> x < 0, SU_Cs .- epsilon)]) *
  " seconds. No iters was " *
  string(findfirst(x -> x < 0, SU_Cs .- epsilon)),
)
