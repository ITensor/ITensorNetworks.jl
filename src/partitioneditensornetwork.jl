using Graphs: dst, src
using ITensors: commoninds
using NamedGraphs.GraphsExtensions: subgraph
using NamedGraphs.PartitionedGraphs: PartitionedGraph, QuotientEdge

function linkinds(pitn::PartitionedGraph, edge::QuotientEdge)
    src_e_itn = subgraph(pitn, src(edge))
    dst_e_itn = subgraph(pitn, dst(edge))
    return commoninds(src_e_itn, dst_e_itn)
end
