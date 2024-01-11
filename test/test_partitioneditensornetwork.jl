using Dictionaries
using Distributions
using GraphsFlows
using ITensors
using ITensorNetworks
using NamedGraphs
using Random
using Test
using SplitApplyCombine

using NamedGraphs: which_partition, parent_graph

@testset "PartitionedITensorNetwork" begin
    g_dims = (1, 6)
    g = named_grid(g_dims)
    s = siteinds("S=1/2", g)
    χ = 4
    Random.seed!(1234)
    ψ = randomITensorNetwork(s; link_space=χ)
    ψψ = ψ ⊗ prime(dag(ψ); sites=[])
    subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ))))
    
    pψψ = PartitionedITensorNetwork(ψψ, subgraph_vertices)

    @show which_partition(pψψ, ((1,1),1))


end