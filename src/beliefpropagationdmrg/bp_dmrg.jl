using NamedGraphs.GraphsExtensions: is_tree
using NamedGraphs.PartitionedGraphs: partitionvertices
using ITensorNetworks: ITensorNetwork, QuadraticFormNetwork, BeliefPropagationCache, update, rescale_messages
using ITensors: scalar

include("utils.jl")
include("bp_extracter.jl")
include("bp_inserter.jl")
include("bp_updater.jl")
include("graphsextensions.jl")

default_bp_update_kwargs(ψ::ITensorNetwork) = is_tree(ψ) ? (;) : (; maxiter = 30, tol = 1e-10)

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

function bp_dmrg(ψ_init::ITensorNetwork, operators::Vector{ITensorNetwork}; no_sweeps = 1, bp_update_kwargs = default_bp_update_kwargs(ψ_init),
    vertex_order_func = ψ -> _bp_region_plan(ψ; nsites = 1, add_additional_traversal = true), energy_calc_fun)

    L = length(vertices(ψ_init))
    state, ψIψ_bpc, ψOψ_bpcs = initialize_caches(ψ_init, operators)
    regions = vertex_order_func(state)

    energy = energy_calc_fun(state, ψIψ_bpc)
    println("Initial energy density is $energy")
    energies = [energy]

    for i in 1:no_sweeps
        println("Beginning sweep $i")
        for region in regions
            region = [rand(vertices(state))]
            println("Updating vertex $region")
            
            v = only(region)
            form_op_v = (v, "operator")
            local_ops = local_op.(ψOψ_bpcs, (v,))
            ψOψ_bpcs = replace_v.(ψOψ_bpcs, (v,))
            ψOψ_bpcs = update.(ψOψ_bpcs; bp_update_kwargs...)
            ψOψ_bpcs_scalars = scalar.(ψOψ_bpcs)
            pv = PartitionVertex(v)
            ψOψ_bpcs = rescale_messages.(ψOψ_bpcs, ψOψ_bpcs_scalars, (pv,))
            ψOψ_bpcs = update_factor.(ψOψ_bpcs, (form_op_v,), local_ops)

            local_state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts = bp_extracter(state, ψOψ_bpcs, ψIψ_bpc, region)

            #Do an eigsolve
            local_state, _ = bp_eigsolve_updater(local_state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts)

            state, ψOψ_bpcs, ψIψ_bpc = bp_inserter(state, ψOψ_bpcs, ψIψ_bpc, local_state, region; cache_update_kwargs =bp_update_kwargs)

            energy = energy_calc_fun(state, ψIψ_bpc)

            println("Current energy density is $energy")
            append!(energies, energy)
        end
    end

    return state, energies
end
