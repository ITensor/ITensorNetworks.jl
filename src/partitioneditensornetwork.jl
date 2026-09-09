using Graphs: dst, src
using ITensors: commoninds
using NamedGraphs.PartitionedGraphs: PartitionedGraph, QuotientEdge
using NamedGraphs: subgraph

function linkinds(pitn::PartitionedGraph, edge::QuotientEdge)
    src_e_itn = subgraph(pitn, src(edge))
    dst_e_itn = subgraph(pitn, dst(edge))
    return commoninds(src_e_itn, dst_e_itn)
end
