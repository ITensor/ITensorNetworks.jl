using ITensors: ITensor, prime

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
        dual_link_index_map = default_dual_link_index_map,
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

function Base.copy(lf::LinearFormNetwork)
    return LinearFormNetwork(
        copy(tensornetwork(lf)), bra_vertex_suffix(lf), ket_vertex_suffix(lf)
    )
end

function update(lf::LinearFormNetwork, original_ket_state_vertex, ket_state::ITensor)
    lf = copy(lf)
    # TODO: Maybe add a check that it really does preserve the graph.
    setindex_preserve_graph!(
        tensornetwork(lf), ket_state, ket_vertex(blf, original_ket_state_vertex)
    )
    return lf
end
