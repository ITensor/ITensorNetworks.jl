using NamedGraphs.GraphsExtensions: is_tree
using NamedGraphs.PartitionedGraphs: partitionvertices
using ITensorNetworks: ITensorNetwork, QuadraticFormNetwork, BeliefPropagationCache, update, rescale_messages
using ITensors: scalar

include("utils.jl")
include("bp_extracter.jl")
include("bp_inserter.jl")
include("bp_updater.jl")

default_bp_update_kwargs(ψ::ITensorNetwork) = is_tree(ψ) ? (;) : (; maxiter = 25, tol = 1e-7)

function initialize_caches(ψ_init::ITensorNetwork, operators::Vector{ITensorNetwork}; cache_update_kwargs = default_bp_update_kwargs(ψ_init))
    ψ = copy(ψ_init)
    ψIψ = QuadraticFormNetwork(ψ)
    ψIψ_bpc = BeliefPropagationCache(ψIψ)
    ψIψ_bpc = update(ψIψ_bpc; cache_update_kwargs...)
    
    ψOψs = QuadraticFormNetwork[QuadraticFormNetwork(operator, ψ) for operator in operators]
    ψOψ_bpcs = BeliefPropagationCache[BeliefPropagationCache(ψOψ) for ψOψ in ψOψs]
    ψOψ_bpcs = BeliefPropagationCache[update(ψOψ_bpc; cache_update_kwargs...) for ψOψ_bpc in ψOψ_bpcs]
    
    return (ψ, ψIψ_bpc, ψOψ_bpcs)
end

function default_vertex_order(ψ_init)
    verts = collect(vertices(ψ_init))
    return [[v] for v in vcat(verts[1:length(verts) - 1], reverse(verts))]
end

alt_vertex_order(ψ_init) = [[v] for v in collect(vertices(ψ_init))]

function bp_dmrg(ψ_init::ITensorNetwork, operators::Vector{ITensorNetwork}; no_sweeps = 1, bp_update_kwargs = default_bp_update_kwargs(ψ_init),
    vertex_order_func = default_vertex_order)

    state, ψIψ_bpc, ψOψ_bpcs = initialize_caches(ψ_init, operators)
    regions = vertex_order_func(state)

    energy = sum(scalar.(ψOψ_bpcs)) / scalar(ψIψ_bpc)
    println("Initial energy is $energy")
    energies = [energy]

    for i in 1:no_sweeps
        println("Beginning sweep $i")
        for region in regions
            println("Updating vertex $region")
            
            v = only(region)
            form_op_v = (v, "operator")
            ψIψ_bpc = update(ψIψ_bpc; bp_update_kwargs...)
            local_ops = local_op.(ψOψ_bpcs, (v,))
            ψOψ_bpcs = replace_v.(ψOψ_bpcs, (v,))
            ψOψ_bpcs = update.(ψOψ_bpcs; bp_update_kwargs...)
            ψOψ_bpcs_scalars = scalar.(ψOψ_bpcs)
            pv = PartitionVertex(v)
            ψOψ_bpcs = BeliefPropagationCache[rescale_messages(ψOψ_bpc, ψOψ_bpc_scalar, pv) for (ψOψ_bpc, ψOψ_bpc_scalar) in zip(ψOψ_bpcs, ψOψ_bpcs_scalars)]
            ψOψ_bpcs = BeliefPropagationCache[update_factor(ψOψ_bpc, form_op_v, local_op) for (ψOψ_bpc, local_op) in zip(ψOψ_bpcs, local_ops)]

            local_state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts = bp_extracter(state, ψOψ_bpcs, ψIψ_bpc, region)

            #Do an eigsolve
            local_state, _ = bp_eigsolve_updater(local_state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts)

            state, ψOψ_bpcs, ψIψ_bpc = bp_inserter(state, ψOψ_bpcs, ψIψ_bpc, local_state, region; cache_update_kwargs =bp_update_kwargs)

            energy = sum(scalar.(ψOψ_bpcs)) / scalar(ψIψ_bpc)
            println("Current energy is $energy")
            append!(energies, energy)
        end
    end

    return state, energies
end
