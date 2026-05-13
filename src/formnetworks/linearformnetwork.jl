using DataGraphs: DataGraphs, set_vertex_data!
using Dictionaries: Dictionary, set!
using ITensors: ITensor, Index, commoninds, dag, prime
using NamedGraphs.GraphsExtensions: disjoint_union
using NamedGraphs.PartitionedGraphs:
    PartitionedGraph, QuotientEdge, partitioned_vertices, quotientedges

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

# Initial BP messages on each quotient edge built from the bra↔ket leg
# pairs crossing that edge: legs are taken from the `from`-partition side
# of each layer (so the bra leg and the ket leg carry opposite directions
# because the bra layer is `dag(dual_link_index_map(ket))`), and the
# forward/reverse messages use opposite-end views so each one's open
# legs face the correct receiving partition when read during BP updates.
# Iteration is over Cartesian products of the original ket-graph vertices
# in each partition, so this works for arbitrary partitionings (per-vertex
# or coarser groupings such as whole columns).
function identity_messages(fn::LinearFormNetwork, ptn::PartitionedGraph)
    pairings = Dictionary{QuotientEdge, Pair{Vector{Index}, Vector{Index}}}()
    tn = tensornetwork(fn)
    pv = partitioned_vertices(ptn)
    ket_s = ket_vertex_suffix(fn)
    for pe in quotientedges(ptn)
        src_orig = unique(first.(filter(v -> last(v) == ket_s, pv[parent(src(pe))])))
        dst_orig = unique(first.(filter(v -> last(v) == ket_s, pv[parent(dst(pe))])))
        for (from_orig, to_orig, e) in (
                (src_orig, dst_orig, pe),
                (dst_orig, src_orig, reverse(pe)),
            )
            bras = Index[]
            kets = Index[]
            for v_from in from_orig, v_to in to_orig
                append!(
                    bras,
                    commoninds(tn[bra_vertex(fn, v_from)], tn[bra_vertex(fn, v_to)])
                )
                append!(
                    kets,
                    commoninds(tn[ket_vertex(fn, v_from)], tn[ket_vertex(fn, v_to)])
                )
            end
            set!(pairings, e, bras => kets)
        end
    end
    return identity_messages(scalartype(tn), pairings)
end
