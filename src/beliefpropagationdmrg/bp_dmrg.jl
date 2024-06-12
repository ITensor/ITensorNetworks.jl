using NamedGraphs.GraphsExtensions: is_tree
using NamedGraphs.PartitionedGraphs: partitionvertices, partitionedges, PartitionEdge
using ITensorNetworks: ITensorNetwork, QuadraticFormNetwork, BeliefPropagationCache, update, default_message_update
using ITensors: scalar

include("utils.jl")
include("bp_extracter.jl")
include("bp_inserter.jl")
include("bp_updater.jl")
include("graphsextensions.jl")

default_bp_update_kwargs(ψ::ITensorNetwork) = is_tree(ψ) ? (;) : (; maxiter = 30, tol = 1e-10)

message_update_f = tns -> default_message_update(tns; normalize = false)

function initialize_caches(ψ_init::ITensorNetwork, operators::Vector{ITensorNetwork}; cache_update_kwargs = default_bp_update_kwargs(ψ_init))
    ψ = copy(ψ_init)
    ψIψ = QuadraticFormNetwork(ψ)
    ψIψ_bpc = BeliefPropagationCache(ψIψ)
    
    ψOψs = QuadraticFormNetwork[QuadraticFormNetwork(operator, ψ) for operator in operators]
    ψOψ_bpcs = BeliefPropagationCache[BeliefPropagationCache(ψOψ) for ψOψ in ψOψs]
    return (ψ, ψIψ_bpc, ψOψ_bpcs)
end

function reduced_bp(state::ITensorNetwork, operators::Vector{ITensorNetwork}, ψIψ_bpc::BeliefPropagationCache,
    ψOψ_bpcs::Vector{<:BeliefPropagationCache}, v)

    ψIψ_bpc_mts = messages(ψIψ_bpc)
    ψOψ_bpcs = copy.(ψOψ_bpcs)

    for (ψOψ_bpc, operator) in zip(ψOψ_bpcs, operators)
        ptn = partitioned_tensornetwork(ψOψ_bpc)
        edge_seq = post_order_dfs_edges(underlying_graph(operator), v)
        broken_edges = setdiff(edges(state), edges(operator))
        partition_broken_edges = PartitionEdge.(broken_edges)
        partition_broken_edges = vcat(partition_broken_edges, reverse.(partition_broken_edges))
        mts = messages(ψOψ_bpc)
        for pe in partition_broken_edges
            set!(mts, pe, copy(ψIψ_bpc_mts[pe]))
        end
        partition_edge_seq = PartitionEdge.(edge_seq)
        ψOψ_bpc = update(ψOψ_bpc, partition_edge_seq; message_update = message_update_f)
    end

    return state, ψOψ_bpcs, ψIψ_bpc 
end


function bp_dmrg(ψ_init::ITensorNetwork, operators::Vector; no_sweeps = 1, bp_update_kwargs = default_bp_update_kwargs(ψ_init),
    vertex_order_func = ψ -> _bp_region_plan(ψ; nsites = 1, add_additional_traversal = true), energy_calc_fun)

    L = length(vertices(ψ_init))
    state, ψIψ_bpc, ψOψ_bpcs = initialize_caches(ψ_init, operators)
    regions = vertex_order_func(state)

    state, ψIψ_bpc, ψOψ_bpcs = updater(state, ψIψ_bpc, ψOψ_bpcs; cache_update_kwargs = bp_update_kwargs)

    energy = energy_calc_fun(state, ψIψ_bpc)
    println("Initial energy density is $energy")
    energies = [energy]
    no_sweeps = 1

    for i in 1:no_sweeps
        println("Beginning sweep $i")
        for region in regions
            println("Updating vertex $region")
            
            v = only(region)
            #TODO: Do BP on ψOψs but default the messages to be incoming to that region to be those from ψIψ_bpc
            state, ψOψ_bpcs, ψIψ_bpc  = reduced_bp(state, operators, ψIψ_bpc, ψOψ_bpcs, v)

            local_state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts = bp_extracter(state, ψOψ_bpcs, ψIψ_bpc, region)

            #Do an eigsolve
            local_state, _ = bp_eigsolve_updater(local_state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts)

            state, ψOψ_bpcs, ψIψ_bpc = bp_inserter(state, ψOψ_bpcs, ψIψ_bpc, local_state, region)

            state, ψIψ_bpc, ψOψ_bpcs = updater(state, ψIψ_bpc, ψOψ_bpcs; cache_update_kwargs = bp_update_kwargs)

            energy = energy_calc_fun(state, ψIψ_bpc)

            println("Current energy density is $energy")
            append!(energies, energy)
        end
    end

    return state, energies
end
