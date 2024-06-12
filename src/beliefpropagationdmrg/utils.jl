using ITensors: siteinds, Op, prime, OpSum, apply
using Graphs: AbstractGraph, SimpleGraph, edges, vertices, is_tree, connected_components
using NamedGraphs: NamedGraph, NamedEdge, NamedGraphs, rename_vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.GraphsExtensions: decorate_graph_edges, forest_cover, add_edges, rem_edges, add_vertices, rem_vertices, disjoint_union, subgraph, src, dst
using NamedGraphs.PartitionedGraphs: PartitionVertex, partitionedge, unpartitioned_graph
using ITensorNetworks: BeliefPropagationCache, AbstractITensorNetwork, AbstractFormNetwork, IndsNetwork, ITensorNetwork, insert_linkinds, ttn, union_all_inds,
    neighbor_vertices, environment, messages, update_factor, message, partitioned_tensornetwork, bra_vertex, ket_vertex, operator_vertex, default_cache_update_kwargs,
    dual_index_map, region_scalar, renormalize_messages, scalar_factors_quotient
using DataGraphs: underlying_graph
using ITensorNetworks.ModelHamiltonians: heisenberg
using ITensors: ITensor, noprime, dag, noncommonind, commonind, replaceind, dim, noncommoninds, delta, replaceinds
using ITensors.NDTensors: denseblocks

function exact_energy(g::AbstractGraph, bpc::BeliefPropagationCache)
    tn = ITensorNetwork(g)
    for v in vertices(g)
        tn[v] =  get_local_term(bpc, v)
    end
    degree_two_sites = filter(v -> degree(tn, v) == 2, vertices(tn))
    while !isempty(degree_two_sites)
        v = first(degree_two_sites)
        vn = first(neighbors(g, v))
        tn = contract(tn, NamedEdge(v => vn); merged_vertex=vn)
        degree_two_sites = filter(v -> degree(tn, v) == 2, vertices(tn))
    end
    return ITensors.contract(ITensor[tn[v] for v in vertices(tn)]; sequence = "automatic")[]
end

function renamer(g)
    vertex_rename = Dictionary()
    for (i,v) in enumerate(vertices(g))
        set!(vertex_rename, v, (i,))
    end
    return rename_vertices(v -> vertex_rename[v], g)
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


