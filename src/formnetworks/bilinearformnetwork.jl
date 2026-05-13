using Adapt: adapt
using DataGraphs: DataGraphs, set_vertex_data!
using Dictionaries: Dictionary, set!
using ITensors.NDTensors: datatype, denseblocks
using ITensors: ITensor, Index, Op, commoninds, dag, delta, prime, sim
using NamedGraphs.GraphsExtensions: disjoint_union
using NamedGraphs.PartitionedGraphs:
    PartitionedGraph, QuotientEdge, partitioned_vertices, quotientedges

default_dual_site_index_map = prime
default_dual_link_index_map = sim

struct BilinearFormNetwork{
        V,
        TensorNetwork <: AbstractITensorNetwork{V},
        OperatorVertexSuffix,
        BraVertexSuffix,
        KetVertexSuffix,
    } <: AbstractFormNetwork{V}
    tensornetwork::TensorNetwork
    operator_vertex_suffix::OperatorVertexSuffix
    bra_vertex_suffix::BraVertexSuffix
    ket_vertex_suffix::KetVertexSuffix
end

function BilinearFormNetwork(
        operator::AbstractITensorNetwork,
        bra::AbstractITensorNetwork,
        ket::AbstractITensorNetwork;
        operator_vertex_suffix = default_operator_vertex_suffix(),
        bra_vertex_suffix = default_bra_vertex_suffix(),
        ket_vertex_suffix = default_ket_vertex_suffix(),
        dual_site_index_map = default_dual_site_index_map,
        dual_link_index_map = default_dual_link_index_map
    )
    bra_mapped = dual_link_index_map(dual_site_index_map(bra; links = []); sites = [])
    tn = disjoint_union(
        operator_vertex_suffix => operator,
        bra_vertex_suffix => dag(bra_mapped),
        ket_vertex_suffix => ket
    )
    return BilinearFormNetwork(
        tn, operator_vertex_suffix, bra_vertex_suffix, ket_vertex_suffix
    )
end

operator_vertex_suffix(blf::BilinearFormNetwork) = blf.operator_vertex_suffix
bra_vertex_suffix(blf::BilinearFormNetwork) = blf.bra_vertex_suffix
ket_vertex_suffix(blf::BilinearFormNetwork) = blf.ket_vertex_suffix
# TODO: Use `NamedGraphs.GraphsExtensions.parent_graph`.
tensornetwork(blf::BilinearFormNetwork) = blf.tensornetwork

# Forward vertex writes to the wrapped network so reverse-index map and
# edge reconciliation run on the underlying `ITensorNetwork`.
function DataGraphs.set_vertex_data!(blf::BilinearFormNetwork, value, vertex)
    set_vertex_data!(tensornetwork(blf), value, vertex)
    return blf
end

function Base.copy(blf::BilinearFormNetwork)
    return BilinearFormNetwork(
        copy(tensornetwork(blf)),
        operator_vertex_suffix(blf),
        bra_vertex_suffix(blf),
        ket_vertex_suffix(blf)
    )
end

function itensor_identity_map(elt::Type, i_pairs::Vector)
    return prod(i_pairs; init = ITensor(one(Bool))) do i_pair
        return denseblocks(delta(elt, last(i_pair), dag(first(i_pair))))
    end
end

itensor_identity_map(i_pairs::Vector) = itensor_identity_map(Float64, i_pairs)

function BilinearFormNetwork(
        bra::AbstractITensorNetwork,
        ket::AbstractITensorNetwork;
        dual_site_index_map = default_dual_site_index_map,
        kwargs...
    )
    bra_site_inds = mapreduce(v -> siteinds(bra, v), vcat, vertices(bra); init = Index[])
    ket_site_inds = mapreduce(v -> siteinds(ket, v), vcat, vertices(ket); init = Index[])
    @assert issetequal(bra_site_inds, ket_site_inds)
    s = siteinds(ket)
    s_mapped = dual_site_index_map(s)
    operator_inds = union_all_inds(s, s_mapped)

    ts = Dict{vertextype(operator_inds), ITensor}()
    for v in vertices(operator_inds)
        ts[v] = itensor_identity_map(scalartype(ket), s[v] .=> s_mapped[v])
    end
    O = ITensorNetwork(ts)
    O = adapt(promote_type(datatype(bra), datatype(ket)), O)
    return BilinearFormNetwork(O, bra, ket; dual_site_index_map, kwargs...)
end

function update(
        blf::BilinearFormNetwork,
        original_bra_state_vertex,
        original_ket_state_vertex,
        bra_state::ITensor,
        ket_state::ITensor
    )
    blf = copy(blf)
    tensornetwork(blf)[bra_vertex(blf, original_bra_state_vertex)] = bra_state
    tensornetwork(blf)[ket_vertex(blf, original_ket_state_vertex)] = ket_state
    return blf
end

# Initial BP messages from bra↔ket pairings on each quotient edge.
# Errors when the operator subnet has its own inter-vertex links (the
# multi-site-operator case): those legs are operator-internal and have
# no bra/ket pair, so no canonical identity initialization exists — the
# caller must supply `messages` explicitly.
function identity_messages(fn::BilinearFormNetwork, ptn::PartitionedGraph)
    pairings = Dictionary{QuotientEdge, Pair{Vector{Index}, Vector{Index}}}()
    tn = tensornetwork(fn)
    pv = partitioned_vertices(ptn)
    ket_s = ket_vertex_suffix(fn)
    for pe in quotientedges(ptn)
        src_orig = unique(first.(filter(v -> last(v) == ket_s, pv[parent(src(pe))])))
        dst_orig = unique(first.(filter(v -> last(v) == ket_s, pv[parent(dst(pe))])))
        for v_from in src_orig, v_to in dst_orig
            op_inds = commoninds(
                tn[operator_vertex(fn, v_from)], tn[operator_vertex(fn, v_to)]
            )
            if !isempty(op_inds)
                error(
                    "BilinearFormNetwork: operator-internal cross-Index between " *
                        "$v_from and $v_to has no bra/ket pair; supply `messages` " *
                        "explicitly to BP."
                )
            end
        end
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
