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

default_solver_algorithm() = "orthogonalize"
default_solver_kwargs() = (; niters = 25, nsites = 1, tolerance = 1e-10, normalize = true, maxdim = nothing, cutoff = nothing)

function blf_update_sequence(g::AbstractGraph; nsites::Int64=1)
    if nsites == 1 || nsites == 2
        es = post_order_dfs_edges(g, first(leaf_vertices(g)))
        vs = [[src(e), dst(e)] for e in es]
        regions = nsites == 2 ? vs : [[v] for v in unique(reduce(vcat, vs))]
        return vcat(regions, reverse(reverse.(regions)))
    else
        error("Nsites > 2 sequences not currently supported")
    end
end

function blf_updater(alg::Algorithm"orthogonalize", xAy_bpc::AbstractBeliefPropagationCache, y::AbstractITensorNetwork, prev_region::Vector, region::Vector)
    path = gauge_path(y, prev_region, region)
    y = gauge_walk(alg, y, path)
    verts = unique(vcat(src.(path), dst.(path)))
    factors = [dag(y[v]) for v in verts]
    xAy_bpc = update_factors(xAy_bpc, Dictionary([(v, "ket") for v in verts], factors))
    pe_path = partitionedges(partitioned_tensornetwork(xAy_bpc), [NamedEdge((src(e), "ket") => (dst(e), "ket")) for e in path])
    xAy_bpc = update(Algorithm("bp"), xAy_bpc, pe_path; message_update_function_kwargs = (; normalize = false))
    return xAy_bpc, y
end

function blf_extracter(xAy_bpc::AbstractBeliefPropagationCache, region::Vector)
    return environment(xAy_bpc, [(v, "ket") for v in region])
end

function blf_inserter(∂xAy_bpc_∂r::Vector{ITensor}, xAy_bpc::AbstractBeliefPropagationCache, y::AbstractITensorNetwork, region::Vector; normalize, maxdim, cutoff)
    yr = contract(∂xAy_bpc_∂r; sequence = "automatic")
    if length(region) == 1
        v = only(region)
        if normalize
            yr /= sqrt(norm_sqr(yr))
        end
        y[v] = yr
    elseif length(region) == 2
        v1, v2 = first(region), last(region)
        linds, cind = uniqueinds(y[v1], y[v2]), commonind(y[v1], y[v2])
        yv1, yv2 = factorize(yr, linds; ortho = "left", tags=tags(cind), cutoff, maxdim)
        if normalize
            yv2 /= sqrt(norm_sqr(yv2))
        end
        y[v1], y[v2] = yv1, yv2
    else
        error("Updates with regions bigger than 2 not currently supported")
    end
    vertices = [(v, "ket") for v in region]
    factors = [y[v] for v in region]
    xAy_bpc = update_factors(xAy_bpc, Dictionary(vertices, factors))
    return y, xAy_bpc
end

function blf_costfunction(xAy::AbstractBeliefPropagationCache, region)
    verts =  [(v, "ket") for v in region]
    return contract([environment(xAy, verts); factors(xAy, verts)]; sequence = "automatic")[]
end

#Optimize over y to maximize <x|A|y> * <y|dag(A)|x> / <y|y> based on a designated partitioning of the bilinearform
function maximize_bilinearform(
    alg::Algorithm"orthogonalize",
    xAy::BilinearFormNetwork,
    y::ITensorNetwork = dag(ket_network(xAy)),
    partition_verts = group(v -> first(v), vertices(xAy)); 
    updater = blf_updater,
    extracter = blf_extracter,
    inserter = blf_inserter,
    costfunction = blf_costfunction,
    sequence = blf_update_sequence,
    normalize::Bool = true,
    niters::Int64 = 25, 
    nsites::Int64 = 1,
    tolerance = nothing,
    maxdim = nothing,
    cutoff = nothing)

    xAy_bpc = BeliefPropagationCache(xAy, partition_verts)
    seq = sequence(y; nsites)

    prev_region = collect(vertices(y))
    cs = zeros(ComplexF64, (niters, length(seq)))
    for i in 1:niters
        for (j, region) in enumerate(seq)
            xAy_bpc, y = updater(alg, xAy_bpc, y, prev_region, region)
            ∂xAy_bpc_∂r = extracter(xAy_bpc, region)
            y, xAy_bpc = inserter(∂xAy_bpc_∂r, xAy_bpc, y, region; normalize, maxdim, cutoff)
            cs[i, j] = costfunction(xAy_bpc, region)
            prev_region = region
        end
        if i >= 2 && (abs(sum(cs[i, :]) - sum(cs[i-1, :]))) / length(seq) <= tolerance
            return dag(y)
        end
    end
    
    return dag(y)
end

function Base.truncate(x::AbstractITensorNetwork; maxdim_init::Int64, kwargs...)
    y = ITensorNetwork(v -> inds -> delta(inds), siteinds(x); link_space = maxdim_init)
    xIy = BilinearFormNetwork(x, y)
    return maximize_bilinearform(xIy, y; kwargs...)
end

function ITensors.apply(A::AbstractITensorNetwork, x::AbstractITensorNetwork; maxdim_init::Int64, kwargs...)
    y = ITensorNetwork(v -> inds -> delta(inds), siteinds(x); link_space = maxdim_init)
    xAy = BilinearFormNetwork(A, x, y)
    return maximize_bilinearform(xAy, y; kwargs...)
end

function maximize_bilinearform(xAy::BilinearFormNetwork, args...; alg = default_solver_algorithm(), solver_kwargs = default_solver_kwargs())
    return maximize_bilinearform(Algorithm(alg), xAy, args...; solver_kwargs...)
end

