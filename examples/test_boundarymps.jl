using ITensorNetworks: BoundaryMPSCache, BeliefPropagationCache, QuadraticFormNetwork, IndsNetwork, siteinds, ttn, random_tensornetwork,
    partitionedplanargraph, gauges, partitionedges, messages, update, partition_update
using ITensorNetworks.ModelHamiltonians: ising
using ITensors: Index, OpSum, terms, sites
using NamedGraphs: NamedEdge
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.GraphsExtensions: rem_vertex
using NamedGraphs.PartitionedGraphs: partitioned_graph, PartitionVertex, PartitionEdge

L = 3
g = named_grid((L,L))
s = siteinds("S=1/2", g)
ψ = random_tensornetwork(s; link_space = 2)
ψIψ = BeliefPropagationCache(QuadraticFormNetwork(ψ))
ψIψ = BoundaryMPSCache(ψIψ)

#@show PartitionEdge(NamedEdge(1 => 2))
#@show PartitionEdge(1 => 2)
@show partition_update(ψIψ, (1,1), (1,2))