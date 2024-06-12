using ITensors: siteinds, Op, prime, OpSum, apply
using Graphs: AbstractGraph, SimpleGraph, edges, vertices, is_tree, connected_components
using NamedGraphs: NamedGraph, NamedEdge, NamedGraphs, rename_vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.GraphsExtensions: decorate_graph_edges, forest_cover, add_edges, rem_edges, add_vertices, rem_vertices, disjoint_union, subgraph, src, dst
using NamedGraphs.PartitionedGraphs: PartitionVertex, partitionedge, unpartitioned_graph
using ITensorNetworks: BeliefPropagationCache, AbstractITensorNetwork, AbstractFormNetwork, IndsNetwork, ITensorNetwork, insert_linkinds, ttn, union_all_inds,
    neighbor_vertices, environment, messages, update_factor, message, partitioned_tensornetwork, bra_vertex, ket_vertex, operator_vertex, default_cache_update_kwargs,
    dual_index_map, region_scalar, renormalize_messages, scalar_factors_quotient, norm_sqr_network
using DataGraphs: underlying_graph
using ITensorNetworks.ModelHamiltonians: heisenberg
using ITensors: ITensor, noprime, dag, noncommonind, commonind, replaceind, dim, noncommoninds, delta, replaceinds, Trotter
using ITensors.NDTensors: denseblocks
using SplitApplyCombine: group

function get_local_term(bpc::BeliefPropagationCache, v)
    qf = tensornetwork(bpc)
    return qf[(v, "ket")]*qf[(v, "operator")]*qf[(v, "bra")]
end

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

function imaginary_time_evo(s::IndsNetwork, ψ::ITensorNetwork, model::Function, dbetas::Vector{<:Tuple}; model_params,
    bp_update_kwargs = (; maxiter = 10, tol = 1e-10), apply_kwargs = (; cutoff = 1e-12, maxdim = 10))
    ψ = copy(ψ)
    g = underlying_graph(ψ)
    
    ℋ =model(g; model_params...)
    ψψ = norm_sqr_network(ψ)
    bpc = BeliefPropagationCache(ψψ, group(v->v[1], vertices(ψψ)))
    bpc = update(bpc; bp_update_kwargs...)
    L = length(vertices(ψ))
    println("Starting Imaginary Time Evolution")
    β = 0
    for (i, period) in enumerate(dbetas)
        nbetas, dβ = first(period), last(period)
        println("Entering evolution period $i , β = $β, dβ = $dβ")
        U = exp(- dβ * ℋ, alg = Trotter{1}())
        gates = Vector{ITensor}(U, s)
        for i in 1:nbetas
            for gate in gates
                ψ, bpc = BP_apply(gate, ψ, bpc; apply_kwargs...)
            end
            β += dβ
            bpc = update(bpc; bp_update_kwargs...)
        end
        e = sum(expect(ψ, ℋ; alg = "bp"))
        println("Energy is $(e / L)")
    end

    return ψ
end


