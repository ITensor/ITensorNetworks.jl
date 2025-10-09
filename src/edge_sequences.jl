using Graphs: IsDirected, connected_components, edges, edgetype
using ITensors.NDTensors: Algorithm, @Algorithm_str
using NamedGraphs: NamedGraphs
using NamedGraphs.GraphsExtensions: GraphsExtensions, forest_cover, undirected_graph
using NamedGraphs.PartitionedGraphs: PartitionEdge, PartitionedGraph, partitions_graph
using SimpleTraits: SimpleTraits, Not, @traitfn

default_edge_sequence_alg() = "forest_cover"
function default_edge_sequence(pg::PartitionedGraph)
    return PartitionEdge.(edge_sequence(partitions_graph(pg)))
end

@traitfn function edge_sequence(
        g::::(!IsDirected); alg = default_edge_sequence_alg(), kwargs...
    )
    return edge_sequence(Algorithm(alg), g; kwargs...)
end

@traitfn function edge_sequence(g::::IsDirected; alg = default_edge_sequence_alg(), kwargs...)
    return edge_sequence(Algorithm(alg), undirected_graph(underlying_graph(g)); kwargs...)
end

@traitfn function edge_sequence(alg::Algorithm, g::::IsDirected; kwargs...)
    return edge_sequence(alg, undirected_graph(underlying_graph(g)); kwargs...)
end

@traitfn function edge_sequence(
        ::Algorithm"forest_cover",
        g::::(!IsDirected);
        root_vertex = GraphsExtensions.default_root_vertex,
    )
    forests = forest_cover(g)
    edges = edgetype(g)[]
    for forest in forests
        trees = [forest[vs] for vs in connected_components(forest)]
        for tree in trees
            tree_edges = post_order_dfs_edges(tree, root_vertex(tree))
            push!(edges, vcat(tree_edges, reverse(reverse.(tree_edges)))...)
        end
    end
    return edges
end

@traitfn function edge_sequence(::Algorithm"parallel", g::::(!IsDirected))
    return [[e] for e in vcat(edges(g), reverse.(edges(g)))]
end
