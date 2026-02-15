using Graphs.SimpleGraphs: SimpleGraph, SimpleGraphs
using ITensors: ITensor, hascommoninds

function SimpleGraphs.SimpleGraph(itensors::Vector{ITensor})
    nv_graph = length(itensors)
    graph = SimpleGraph(nv_graph)
    for i in 1:(nv_graph - 1), j in (i + 1):nv_graph
        if hascommoninds(itensors[i], itensors[j])
            add_edge!(graph, i => j)
        end
    end
    return graph
end
