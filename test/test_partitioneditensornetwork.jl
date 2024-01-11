using Dictionaries
using Distributions
using GraphsFlows
using ITensors
using ITensorNetworks
using NamedGraphs
using Random
using Test
using SplitApplyCombine

using NamedGraphs: which_partition, parent_graph, partitioned_vertices

using ITensorNetworks: message_tensors, update_message_tensor, belief_propagation_iteration, belief_propagation,
    get_environment

@testset "PartitionedITensorNetwork" begin
    g_dims = (3, 3)
    g = named_grid(g_dims)
    s = siteinds("S=1/2", g)
    χ = 2
    Random.seed!(1234)
    ψ = randomITensorNetwork(s; link_space=χ)
    ψψ = ψ ⊗ prime(dag(ψ); sites=[])
    subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ))))
    
    pψψ = PartitionedITensorNetwork(ψψ, subgraph_vertices)

    mts = message_tensors(pψψ)
    mts = belief_propagation(pψψ, mts; contract_kwargs=(; alg="exact"), verbose = true, niters = 10, target_precision = 1e-3)

    env_tensors = get_environment(pψψ, mts, [((1,1),1)])

    @show ITensors.contract(env_tensors)

end