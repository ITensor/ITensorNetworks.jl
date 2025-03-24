using ITensorNetworks: ITensorNetworks, ITensorNetwork, random_tensornetwork, siteinds, QuadraticFormNetwork, prime,
    BilinearFormNetwork, environment, subgraph, BeliefPropagationCache, partitioned_graph,
    ket_network, tree_gauge, tree_orthogonalize, gauge_path, gauge_walk, update_factors, partitioned_tensornetwork,
    update, update_factor, region_scalar
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs: NamedEdge
using TensorOperations: TensorOperations
using ITensors: ITensors, ITensor, contract, dag
using Graphs: is_tree
using NamedGraphs.PartitionedGraphs: partitioned_graph, partitionedges, partitionvertex, partitionvertices
using NamedGraphs.GraphsExtensions: bfs_tree, leaf_vertices, post_order_dfs_edges, src, dst, vertices
using NDTensors: Algorithm
using Dictionaries
using LinearAlgebra: norm_sqr

function update_sequence(tn::ITensorNetwork; nsites::Int64=1)
    @assert is_tree(tn)
    if nsites == 1 || nsites == 2
        es = post_order_dfs_edges(tn, first(leaf_vertices(tn)))
        vs = [[src(e), dst(e)] for e in es]
        nsites == 2 && return vs
        nsites == 1 && return [[v] for v in unique(reduce(vcat, vs))]
    else
        error("Nsites > 2 sequences not currently supported")
    end
end

function default_fit_updater(xAy_bpc::BeliefPropagationCache, y::ITensorNetwork, prev_region::Vector, region::Vector)
    path = gauge_path(y, prev_region, region)
    y = gauge_walk(Algorithm("orthogonalize"), y, path)
    verts = unique(vcat(src.(path), dst.(path)))
    factors = [dag(y[v]) for v in verts]
    xAy_bpc = update_factors(xAy_bpc, Dictionary([(v, "ket") for v in verts], factors))
    pe_path = partitionedges(partitioned_tensornetwork(xAy_bpc), [NamedEdge((src(e), "ket") => (dst(e), "ket")) for e in path])
    xAy_bpc = update(Algorithm("bp"), xAy_bpc, pe_path; message_update_function_kwargs = (; normalize = false))
    return xAy_bpc, y
end

function default_fit_extracter(xAy_bpc::BeliefPropagationCache, region::Vector)
    return environment(xAy_bpc, [(v, "ket") for v in region])
end

function default_fit_inserter(∂xAy_bpc_∂r::Vector{ITensor}, xAy_bpc::BeliefPropagationCache, y::ITensorNetwork, region::Vector;
    normalize = true)
    if length(region) == 1
        v = only(region)
        yv = contract(∂xAy_bpc_∂r; sequence = "automatic")
        if normalize
            yv /= sqrt(norm_sqr(yv))
        end
        y[v] = yv
        xAy_bpc = update_factor(xAy_bpc, (v, "ket"), dag(yv))
    else
        error("Updates with regions bigger than 1 not supported")
    end
    return y, xAy_bpc
end

function default_costfunction(xAy::BeliefPropagationCache, region)
    if length(region) == 1
        pv = only(partitionvertices(xAy, [(only(region), "ket")]))
        c = region_scalar(xAy, pv)
        return sqrt(c * conj(c))
    else
        error("Cost Functions with regions bigger than 1 not supported")
    end
end


function fit_ket(xAy::BilinearFormNetwork, partition_verts::Vector; 
    updater = default_fit_updater,
    extracter = default_fit_extracter,
    inserter = default_fit_inserter,
    costfunction = default_costfunction,
    normalize = true,
    niters::Int64 = 10, nsites::Int64=1)
    xAy_bpc = BeliefPropagationCache(xAy, partition_verts)
    y = dag(ket_network(xAy))
    @assert is_tree(partitioned_graph(xAy_bpc))
    seq = update_sequence(y; nsites)

    for i in 1:niters
        prev_region = collect(vertices(y))
        for region in seq
            xAy_bpc, y = updater(xAy_bpc, y, prev_region, region)
            ∂xAy_bpc_∂r = extracter(xAy_bpc, region)
            y, xAy_bpc = inserter(∂xAy_bpc_∂r, xAy_bpc, y, region; normalize)
            c = default_costfunction(xAy_bpc, region)
            @show c
            prev_region = region
        end
    end
    
end

ITensors.disable_warn_order()

g = named_grid((3,4))
s = siteinds("S=1/2", g)

a = random_tensornetwork(g; link_space = 2)

a1 = subgraph(a, [(1,1), (1,2), (1,3), (1,4)])
a2 = subgraph(a, [(2,1), (2,2), (2,3), (2,4)])
a3 = subgraph(a, [(3,1), (3,2), (3,3), (3,4)])

c = BilinearFormNetwork(a2, a1, a3; dual_site_index_map = dag, dual_link_index_map = dag)

partition_verts = [[((1,1), "bra"), ((3,1), "ket"), ((2,1), "operator")],
    [((1,2), "bra"), ((3,2), "ket"), ((2,2), "operator")], [((1,3), "bra"), ((3,3), "ket"), ((2,3), "operator")], [((1,4), "bra"), ((3,4), "ket"), ((2,4), "operator")]]

fit_ket(c, partition_verts)

