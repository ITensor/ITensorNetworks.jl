using ITensorNetworks: AbstractITensorNetwork, AbstractBeliefPropagationCache, ITensorNetworks, ITensorNetwork, random_tensornetwork, siteinds, QuadraticFormNetwork, prime,
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

default_fit_algorithm() = "orthogonalize"
default_fit_kwargs() = (; niters = 20, nsites = 1, tolerance = 1e-10, normalize = true)

function default_update_sequence(tn::AbstractITensorNetwork; nsites::Int64=1)
    @assert is_tree(tn)
    if nsites == 1 || nsites == 2
        es = post_order_dfs_edges(tn, first(leaf_vertices(tn)))
        vs = [[src(e), dst(e)] for e in es]
        regions = nsites == 2 ? vs : [[v] for v in unique(reduce(vcat, vs))]
        return vcat(regions, reverse(reverse.(regions)))
    else
        error("Nsites > 2 sequences not currently supported")
    end
end

function default_fit_updater(alg::Algorithm"orthogonalize", xAy_bpc::AbstractBeliefPropagationCache, y::AbstractITensorNetwork, prev_region::Vector, region::Vector)
    path = gauge_path(y, prev_region, region)
    y = gauge_walk(alg, y, path)
    verts = unique(vcat(src.(path), dst.(path)))
    factors = [dag(y[v]) for v in verts]
    xAy_bpc = update_factors(xAy_bpc, Dictionary([(v, "ket") for v in verts], factors))
    pe_path = partitionedges(partitioned_tensornetwork(xAy_bpc), [NamedEdge((src(e), "ket") => (dst(e), "ket")) for e in path])
    xAy_bpc = update(Algorithm("bp"), xAy_bpc, pe_path; message_update_function_kwargs = (; normalize = false))
    return xAy_bpc, y
end

function default_fit_extracter(xAy_bpc::AbstractBeliefPropagationCache, region::Vector)
    return environment(xAy_bpc, [(v, "ket") for v in region])
end

function default_fit_inserter(∂xAy_bpc_∂r::Vector{ITensor}, xAy_bpc::AbstractBeliefPropagationCache, y::AbstractITensorNetwork, region::Vector; normalize = true)
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

function default_costfunction(xAy::AbstractBeliefPropagationCache, region)
    if length(region) == 1
        pv = only(partitionvertices(xAy, [(only(region), "ket")]))
        c = region_scalar(xAy, pv)
        return sqrt(c * conj(c))
    else
        error("Cost Functions with regions bigger than 1 not supported")
    end
end

#Optimize over y to maximize <x|A|y> * <y|dag(A)|x> / <y|y> based on a designated partitioning of the bilinearform
function maximize_bilinearform(
    alg::Algorithm"orthogonalize",
    xAy::BilinearFormNetwork,
    y::ITensorNetwork = dag(ket_network(xAy)),
    partition_verts = group(v -> first(v), vertices(xAy)); 
    updater = default_fit_updater,
    extracter = default_fit_extracter,
    inserter = default_fit_inserter,
    costfunction = default_costfunction,
    sequence = default_update_sequence,
    normalize::Bool,
    niters::Int64, 
    nsites::Int64,
    tolerance::Float64)

    xAy_bpc = BeliefPropagationCache(xAy, partition_verts)
    @assert is_tree(partitioned_graph(xAy_bpc))
    seq = sequence(y; nsites)

    prev_region = collect(vertices(y))
    cs = zeros(ComplexF64, (niters, length(seq)))
    for i in 1:niters
        for (j, region) in enumerate(seq)
            xAy_bpc, y = updater(alg, xAy_bpc, y, prev_region, region)
            ∂xAy_bpc_∂r = extracter(xAy_bpc, region)
            y, xAy_bpc = inserter(∂xAy_bpc_∂r, xAy_bpc, y, region; normalize)
            cs[i, j] = costfunction(xAy_bpc, region)
            prev_region = region
        end
        if i >= 2 && abs(sum(cs[i, :]) - sum(cs[i-1, :])) / length(seq) <= tolerance
            return y
        end
    end
    
    return y
end

function Base.truncate(x::AbstractITensorNetwork; maxdim::Int64, kwargs...)
    y = random_tensornetwork(scalartype(x), siteinds(x); link_space = maxdim)
    xIy = BilinearFormNetwork(x, y)
    return dag(maximize_bilinearform(xIy, y; kwargs...))
end

function ITensors.apply(A::AbstractITensorNetwork, x::AbstractITensorNetwork; maxdim::Int64, kwargs...)
    y = random_tensornetwork(scalartype(x), siteinds(x); link_space = maxdim)
    xAy = BilinearFormNetwork(A, x, y)
    return dag(maximize_bilinearform(xAy, y; kwargs...))
end

function maximize_bilinearform(xAy::BilinearFormNetwork, args...; fit_alg = default_fit_algorithm(), fit_kwargs = default_fit_kwargs())
    return maximize_bilinearform(Algorithm(fit_alg), xAy, args...; fit_kwargs...)
end

