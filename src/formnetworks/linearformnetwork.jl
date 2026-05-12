using DataGraphs: DataGraphs, set_vertex_data!
using Graphs: AbstractGraph
using ITensors: ITensor, prime
using NamedGraphs.GraphsExtensions: disjoint_union
using NamedGraphs: similar_graph

default_dual_link_index_map = prime

struct LinearFormNetwork{
        V, TensorNetwork <: AbstractITensorNetwork{V}, BraVertexSuffix, KetVertexSuffix,
    } <: AbstractFormNetwork{V}
    tensornetwork::TensorNetwork
    bra_vertex_suffix::BraVertexSuffix
    ket_vertex_suffix::KetVertexSuffix
end

function LinearFormNetwork(
        bra::AbstractITensorNetwork,
        ket::AbstractITensorNetwork;
        bra_vertex_suffix = default_bra_vertex_suffix(),
        ket_vertex_suffix = default_ket_vertex_suffix(),
        dual_link_index_map = default_dual_link_index_map
    )
    bra_mapped = dual_link_index_map(bra; sites = [])
    tn = disjoint_union(bra_vertex_suffix => dag(bra_mapped), ket_vertex_suffix => ket)
    return LinearFormNetwork(tn, bra_vertex_suffix, ket_vertex_suffix)
end

function LinearFormNetwork(blf::BilinearFormNetwork)
    bra, ket, operator = subgraph(blf, bra_vertices(blf)),
        subgraph(blf, ket_vertices(blf)),
        subgraph(blf, operator_vertices(blf))
    bra_suffix, ket_suffix = bra_vertex_suffix(blf), ket_vertex_suffix(blf)
    operator = rename_vertices(v -> bra_vertex_map(blf)(v), operator)
    tn = union(bra, ket, operator)
    return LinearFormNetwork(tn, bra_suffix, ket_suffix)
end

bra_vertex_suffix(lf::LinearFormNetwork) = lf.bra_vertex_suffix
ket_vertex_suffix(lf::LinearFormNetwork) = lf.ket_vertex_suffix
# TODO: Use `NamedGraphs.GraphsExtensions.parent_graph`.
tensornetwork(lf::LinearFormNetwork) = lf.tensornetwork

# Forward vertex writes to the wrapped network so reverse-index map and
# edge reconciliation run on the underlying `ITensorNetwork`.
function DataGraphs.set_vertex_data!(lf::LinearFormNetwork, value, vertex)
    set_vertex_data!(tensornetwork(lf), value, vertex)
    return lf
end

function NamedGraphs.similar_graph(
        lf::LinearFormNetwork,
        underlying_graph::AbstractGraph
    )
    tn = similar_graph(tensornetwork(lf), underlying_graph)
    return LinearFormNetwork(tn, bra_vertex_suffix(lf), ket_vertex_suffix(lf))
end

function Base.copy(lf::LinearFormNetwork)
    return LinearFormNetwork(
        copy(tensornetwork(lf)), bra_vertex_suffix(lf), ket_vertex_suffix(lf)
    )
end

function update(lf::LinearFormNetwork, original_ket_state_vertex, ket_state::ITensor)
    lf = copy(lf)
    tensornetwork(lf)[ket_vertex(blf, original_ket_state_vertex)] = ket_state
    return lf
end
