using Graphs: dst, src
using ITensors: commoninds
using NamedGraphs.GraphsExtensions: subgraph
using NamedGraphs.PartitionedGraphs: PartitionedGraph, QuotientEdge

# TODO: Delete this once it is fixed in NamedGraphs.jl.
using Graphs: induced_subgraph
using NamedGraphs.GraphsExtensions: GraphsExtensions
using NamedGraphs.PartitionedGraphs: PartitionedGraph, QuotientVertex
function GraphsExtensions.subgraph(g::PartitionedGraph, v::QuotientVertex)
    return induced_subgraph(g, v)[1].graph
end

function linkinds(pitn::PartitionedGraph, edge::QuotientEdge)
    src_e_itn = subgraph(pitn, src(edge))
    dst_e_itn = subgraph(pitn, dst(edge))
    return commoninds(src_e_itn, dst_e_itn)
end
