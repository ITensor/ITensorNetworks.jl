module ITensorNetworksGraphsFlowsExt
using Graphs: AbstractGraph
using GraphsFlows: GraphsFlows
using ITensorNetworks: ITensorNetworks
using NDTensors.AlgorithmSelection: @Algorithm_str

function ITensorNetworks.mincut(
        ::Algorithm"GraphsFlows",
        graph::AbstractGraph,
        source_vertex,
        target_vertex;
        capacity_matrix,
        alg = GraphsFlows.PushRelabelAlgorithm(),
    )
    # TODO: Replace with `Backend(backend)`.
    return GraphsFlows.mincut(graph, source_vertex, target_vertex, capacity_matrix, alg)
end

end
