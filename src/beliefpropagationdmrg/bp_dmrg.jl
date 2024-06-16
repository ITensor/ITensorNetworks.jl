using NamedGraphs.GraphsExtensions: is_tree
using NamedGraphs.PartitionedGraphs: partitionvertices, partitionedges, PartitionEdge
using ITensorNetworks: ITensorNetwork, QuadraticFormNetwork, BeliefPropagationCache, update, default_message_update
using ITensors: scalar

include("utils.jl")
include("bp_extracter.jl")
include("bp_inserter.jl")
include("bp_updater.jl")

default_bp_update_kwargs(ψ::ITensorNetwork) = is_tree(ψ) ? (;) : (; maxiter = 50, tol = 1e-12)

function initialize_cache(ψ_init::ITensorNetwork; cache_update_kwargs = default_bp_update_kwargs(ψ_init))
    ψ = copy(ψ_init)
    ψIψ = QuadraticFormNetwork(ψ)
    ψIψ_bpc = BeliefPropagationCache(ψIψ)
    state, ψIψ_bpc = renormalize_update_norm_cache(ψ, ψIψ_bpc; cache_update_kwargs)

    return (ψ, ψIψ_bpc)
end

function bp_dmrg(ψ_init::ITensorNetwork, H::OpSum; nsites = 1, no_sweeps = 1, bp_update_kwargs = default_bp_update_kwargs(ψ_init))

    L = length(vertices(ψ_init))
    state, ψIψ_bpc = initialize_cache(ψ_init; cache_update_kwargs = bp_update_kwargs)
    state_vertices, state_edges = collect(vertices(state)), edges(state)
    if nsites == 1
        regions = [[v] for v in vcat(state_vertices, reverse(state_vertices))]
    else 
        regions = [[src(e), dst(e)] for e in vcat(state_edges, reverse(state_edges))]
    end

    init_energy = sum(expect(state, H; alg="bp", (cache!)=Ref(ψIψ_bpc))) / L
    println("Initial energy density is $(init_energy)")
    energies = [init_energy]

    for i in 1:no_sweeps
        println("Beginning sweep $i")
        for region in regions
            println("Updating vertex $region")

            local_state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts = bp_extracter(state, H, ψIψ_bpc, region)

            local_state, energy = bp_eigsolve_updater(local_state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts)

            state, ψIψ_bpc = bp_inserter(state, ψIψ_bpc, local_state, region; bp_update_kwargs)

            append!(energies, energy / L)
            println("Current energy density is $(energy / L)")
        end
    end

    return state, energies
end
