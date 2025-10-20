using Graphs: vertices
using NamedGraphs: AbstractNamedGraph, NamedEdge
using NamedGraphs.PartitionedGraphs: partitionedges
using Printf: @printf

@kwdef mutable struct FittingProblem{State <: AbstractBeliefPropagationCache} <:
    AbstractProblem
    state::State
    ket_graph::AbstractNamedGraph
    overlap::Number = 0
    gauge_region
end

state(F::FittingProblem) = F.state
ket_graph(F::FittingProblem) = F.ket_graph
overlap(F::FittingProblem) = F.overlap
gauge_region(F::FittingProblem) = F.gauge_region

function ket(F::FittingProblem)
    ket_vertices = vertices(ket_graph(F))
    return first(induced_subgraph(tensornetwork(state(F)), ket_vertices))
end

function extract!(region_iter::RegionIterator{<:FittingProblem})
    prob = problem(region_iter)

    region = current_region(region_iter)
    prev_region = gauge_region(prob)
    tn = state(prob)
    path = edge_sequence_between_regions(ket_graph(prob), prev_region, region)
    tn = gauge_walk(Algorithm("orthogonalize"), tn, path)
    pe_path = partitionedges(partitioned_tensornetwork(tn), path)
    tn = update(
        Algorithm("bp"), tn, pe_path; message_update_function_kwargs = (; normalize = false)
    )
    local_tensor = environment(tn, region)
    sequence = contraction_sequence(local_tensor; alg = "optimal")
    local_tensor = dag(contract(local_tensor; sequence))
    #problem, local_tensor = subspace_expand(problem, local_tensor, region; sweep, kws...)

    prob.state = tn
    prob.gauge_region = region

    return region_iter, local_tensor
end

function update!(
        region_iter::RegionIterator{<:FittingProblem}, local_tensor; outputlevel = 0
    )
    F = problem(region_iter)

    region = current_region(region_iter)

    n = (local_tensor * dag(local_tensor))[]
    F.overlap = n / sqrt(n)

    if outputlevel >= 2
        @printf("  Region %s: squared overlap = %.12f\n", region, overlap(F))
    end

    return region_iter, local_tensor
end

function region_plan(F::FittingProblem; nsites, sweep_kwargs...)
    return euler_sweep(ket_graph(F); nsites, sweep_kwargs...)
end

function fit_tensornetwork(
        overlap_network,
        args...;
        nsweeps = 25,
        nsites = 1,
        outputlevel = 0,
        normalize = true,
        factorize_kwargs,
        extra_sweep_kwargs...,
    )
    bpc = BeliefPropagationCache(overlap_network, args...)
    ket_graph = first(
        induced_subgraph(underlying_graph(overlap_network), ket_vertices(overlap_network))
    )
    init_prob = FittingProblem(;
        ket_graph, state = bpc, gauge_region = collect(vertices(ket_graph))
    )

    insert!_kwargs = (; normalize, set_orthogonal_region = false)
    update!_kwargs = (; outputlevel)

    sweep_kwargs = (; nsites, outputlevel, update!_kwargs, insert!_kwargs, factorize_kwargs)
    kwargs_array = [(; sweep_kwargs..., extra_sweep_kwargs..., sweep) for sweep in 1:nsweeps]

    sweep_iter = SweepIterator(init_prob, kwargs_array)
    converged_prob = problem(sweep_solve!(sweep_iter))

    return rename_vertices(inv_vertex_map(overlap_network), ket(converged_prob))
end

function fit_tensornetwork(tn, init_state, args...; kwargs...)
    return fit_tensornetwork(inner_network(tn, init_state), args; kwargs...)
end

#function truncate(tn; maxdim=default_maxdim(), cutoff=default_cutoff(), kwargs...)
#  init_state = ITensorNetwork(
#    v -> inds -> delta(inds), siteinds(tn); link_space=maxdim
#  )
#  overlap_network = inner_network(tn, init_state)
#  insert_kwargs = (; trunc=(; cutoff, maxdim))
#  return fit_tensornetwork(overlap_network; insert_kwargs, kwargs...)
#end

function ITensors.apply(
        A::AbstractITensorNetwork,
        x::AbstractITensorNetwork;
        maxdim = typemax(Int),
        cutoff = 0.0,
        sweep_kwargs...,
    )
    init_state = ITensorNetwork(v -> inds -> delta(inds), siteinds(x); link_space = maxdim)
    overlap_network = inner_network(x, A, init_state)
    return fit_tensornetwork(
        overlap_network; factorize_kwargs = (; maxdim, cutoff), sweep_kwargs...
    )
end
