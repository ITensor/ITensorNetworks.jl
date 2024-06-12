using NamedGraphs.GraphsExtensions: is_tree
using NamedGraphs.PartitionedGraphs: partitionvertices, partitionedges, PartitionEdge
using ITensorNetworks: ITensorNetwork, QuadraticFormNetwork, BeliefPropagationCache, update, default_message_update
using ITensors: scalar

include("utils.jl")
include("bp_extracter.jl")
include("bp_inserter.jl")
include("bp_updater.jl")
include("graphsextensions.jl")

default_bp_update_kwargs(ψ::ITensorNetwork) = is_tree(ψ) ? (;) : (; maxiter = 50, tol = 1e-14)

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
    new_ψOψ_bpcs = BeliefPropagationCache[]

    for (ψOψ_bpc, operator) in zip(ψOψ_bpcs, operators)
        new_ψOψ_bpc = copy(ψOψ_bpc)
        ptn = partitioned_tensornetwork(new_ψOψ_bpc)
        edge_seq = post_order_dfs_edges(underlying_graph(operator), v)
        broken_edges = setdiff(edges(state), edges(operator))
        partition_broken_edges = PartitionEdge.(broken_edges)
        partition_broken_edges = vcat(partition_broken_edges, reverse.(partition_broken_edges))
        mts = messages(new_ψOψ_bpc)
        for pe in partition_broken_edges
            set!(mts, pe, copy(ψIψ_bpc_mts[pe]))
        end
        partition_edge_seq = PartitionEdge.(edge_seq)
        new_ψOψ_bpc = update(new_ψOψ_bpc, partition_edge_seq; message_update = message_update_f)
        push!(new_ψOψ_bpcs, new_ψOψ_bpc)
    end

    return state, new_ψOψ_bpcs, ψIψ_bpc 
end

function updater(ψ::ITensorNetwork, ψIψ_bpc::BeliefPropagationCache, ψOψ_bpcs::Vector{<:BeliefPropagationCache};
    cache_update_kwargs)

    ψ = copy(ψ)
    ψOψ_bpcs = copy.(ψOψ_bpcs)
    ψIψ_bpc = update(ψIψ_bpc; cache_update_kwargs...)
    ψIψ_bpc = renormalize_messages(ψIψ_bpc)
    qf = tensornetwork(ψIψ_bpc)

    for v in vertices(ψ)
        v_ket, v_bra = ket_vertex(qf, v),  bra_vertex(qf, v)
        pv = only(partitionvertices(ψIψ_bpc, [v_ket]))
        vn = region_scalar(ψIψ_bpc, pv)
        state = (1.0 / sqrt(vn)) * ψ[v]
        state_dag = copy(state)
        state_dag = replaceinds(dag(state_dag), inds(state_dag), dual_index_map(qf).(inds(state_dag)))
        vertices_states = Dictionary([v_ket, v_bra], [state, state_dag])
        ψOψ_bpcs = update_factors.(ψOψ_bpcs, (vertices_states,))
        ψIψ_bpc = update_factors(ψIψ_bpc, vertices_states)
        ψ[v] = state
    end

    return ψ, ψIψ_bpc, ψOψ_bpcs
end


function bp_dmrg(ψ_init::ITensorNetwork, operators::Vector; no_sweeps = 1, bp_update_kwargs = default_bp_update_kwargs(ψ_init),
    vertex_order_func = ψ -> bp_region_plan(ψ; nsites = 1, add_additional_traversal = false), energy_calc_fun)

    L = length(vertices(ψ_init))
    state, ψIψ_bpc, ψOψ_bpcs = initialize_caches(ψ_init, operators)
    regions = vertex_order_func(state)

    state, ψIψ_bpc, ψOψ_bpcs = updater(state, ψIψ_bpc, ψOψ_bpcs; cache_update_kwargs = bp_update_kwargs)

    energy = real(energy_calc_fun(state, ψIψ_bpc))
    println("Initial energy density is $(energy)")
    energies = [energy]

    for i in 1:no_sweeps
        println("Beginning sweep $i")
        for region in regions
            println("Updating vertex $region")
        
            v = only(region)
            #v = rand(vertices())
            #TODO: Do BP on ψOψs but default the messages to be incoming to that region to be those from ψIψ_bpc
            state, ψOψ_bpcs, ψIψ_bpc  = reduced_bp(state, operators, ψIψ_bpc, ψOψ_bpcs, v)

            local_state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts = bp_extracter(state, ψOψ_bpcs, ψIψ_bpc, region)

            #Do an eigsolve
            local_state, energy = bp_eigsolve_updater(local_state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts, last(energies); L)

            state, ψOψ_bpcs, ψIψ_bpc = bp_inserter(state, ψOψ_bpcs, ψIψ_bpc, local_state, region)

            state, ψIψ_bpc, ψOψ_bpcs = updater(state, ψIψ_bpc, ψOψ_bpcs; cache_update_kwargs = bp_update_kwargs)

            # if energy < last(energies)
            #     println("Current energy density is $(energy / L)")
            #     append!(energies, energy)
            # else
            #     println("Rejected Move")
            # end


            energy = energy_calc_fun(state, ψIψ_bpc)
            append!(energies, energy)
            println("Current energy density is $(energy)")
        end
    end

    return state, energies
end
