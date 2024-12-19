using ITensorNetworks: BoundaryMPSCache, BeliefPropagationCache, QuadraticFormNetwork, IndsNetwork, siteinds, ttn, random_tensornetwork,
    partitionedges, messages, update, partition_update, set_messages, message,
    planargraph_partitionedges, update_sequence, switch_messages, mps_update, environment, VidalITensorNetwork, ITensorNetwork, expect
using ITensorNetworks.ModelHamiltonians: ising
using ITensors: ITensors, Index, OpSum, terms, sites, contract
using NamedGraphs: NamedEdge
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.GraphsExtensions: rem_vertex
using NamedGraphs.PartitionedGraphs: partitioned_graph, PartitionVertex, PartitionEdge
using LinearAlgebra: normalize
using Graphs: center

using Random

Random.seed!(1234)

L = 4
g = named_grid((L,L))
g = rem_vertex(g, (2,2))
vc = first(center(g))
s = siteinds("S=1/2", g)
ψ = random_tensornetwork(ComplexF64, s; link_space = 4)
bp_update_kwargs = (; maxiter = 50, tol = 1e-14)

#Run BP first to normalize and put in a stable gauge
ψIψ = BeliefPropagationCache(QuadraticFormNetwork(ψ))
ψIψ = update(ψIψ; bp_update_kwargs...)
ψ = VidalITensorNetwork(ψ; cache! = Ref(ψIψ), update_cache = false, cache_update_kwargs = (; maxiter = 0))
ψ = ITensorNetwork(ψ)
ψIψ = BeliefPropagationCache(QuadraticFormNetwork(ψ))

ψIψ = BoundaryMPSCache(ψIψ; sort_f = v -> first(v))

ψIψ = set_messages(ψIψ; message_rank = 4)

ψIψ = mps_update(ψIψ; maxiter = 10, niters = 5)

ψIψ = partition_update(ψIψ, vc)

ρ = contract(environment(ψIψ, [(vc, "operator")]); sequence = "automatic")
sz = contract([ρ, ITensors.op("Z", s[vc])])[] /contract([ρ, ITensors.op("I", s[vc])])[]

@show sz

@show expect(ψ, "Z", [vc]; alg = "bp")