using DataGraphs: DataGraphs, set_vertex_data!, underlying_graph, vertex_data
using Dictionaries: Dictionary, set!
using ITensors: ITensor, commoninds, dag, delta
using NamedGraphs.PartitionedGraphs: PartitionedGraph, QuotientEdge, quotientedges

default_index_map = prime
default_inv_index_map = noprime

struct QuadraticFormNetwork{
        V,
        FormNetwork <: BilinearFormNetwork{V},
        IndexMap,
        InvIndexMap,
    } <:
    AbstractFormNetwork{V}
    formnetwork::FormNetwork
    dual_index_map::IndexMap
    dual_inv_index_map::InvIndexMap
end

bilinear_formnetwork(qf::QuadraticFormNetwork) = qf.formnetwork

#Needed for implementation, forward from bilinear form
for f in [
        :operator_vertex_suffix,
        :bra_vertex_suffix,
        :ket_vertex_suffix,
        :tensornetwork,
    ]
    @eval begin
        function $f(qf::QuadraticFormNetwork, args...; kwargs...)
            return $f(bilinear_formnetwork(qf), args...; kwargs...)
        end
    end
end

function DataGraphs.underlying_graph(qf::QuadraticFormNetwork)
    return underlying_graph(bilinear_formnetwork(qf))
end
DataGraphs.vertex_data(qf::QuadraticFormNetwork) = vertex_data(bilinear_formnetwork(qf))

dual_index_map(qf::QuadraticFormNetwork) = qf.dual_index_map
dual_inv_index_map(qf::QuadraticFormNetwork) = qf.dual_inv_index_map

# Forward vertex writes to the inner `BilinearFormNetwork`, which in turn
# forwards to the underlying `ITensorNetwork`.
function DataGraphs.set_vertex_data!(qf::QuadraticFormNetwork, value, vertex)
    set_vertex_data!(bilinear_formnetwork(qf), value, vertex)
    return qf
end
function Base.copy(qf::QuadraticFormNetwork)
    return QuadraticFormNetwork(
        copy(bilinear_formnetwork(qf)), dual_index_map(qf), dual_inv_index_map(qf)
    )
end

function QuadraticFormNetwork(
        operator::AbstractITensorNetwork,
        ket::AbstractITensorNetwork;
        dual_index_map = default_index_map,
        dual_inv_index_map = default_inv_index_map,
        kwargs...
    )
    blf = BilinearFormNetwork(
        operator,
        ket,
        ket;
        dual_site_index_map = dual_index_map,
        dual_link_index_map = dual_index_map,
        kwargs...
    )
    return QuadraticFormNetwork(blf, dual_index_map, dual_inv_index_map)
end

function QuadraticFormNetwork(
        ket::AbstractITensorNetwork;
        dual_index_map = default_index_map,
        dual_inv_index_map = default_inv_index_map,
        kwargs...
    )
    blf = BilinearFormNetwork(
        ket,
        ket;
        dual_site_index_map = dual_index_map,
        dual_link_index_map = dual_index_map,
        kwargs...
    )
    return QuadraticFormNetwork(blf, dual_index_map, dual_inv_index_map)
end

# Build initial BP messages on each quotient edge as `delta(bra, ket)`
# pairs, one per ket link Index crossing the cut. The bra-side counterpart
# of each ket Index is computed explicitly via `dual_index_map(fn)`, so
# the pairing is correct even when multiple link indices share an edge
# (where `commoninds`-zip ordering between layers is not guaranteed).
function identity_messages(
        fn::QuadraticFormNetwork;
        partitioned_vertices = default_partitioned_vertices(fn)
    )
    ptn = PartitionedGraph(fn, partitioned_vertices)
    messages = Dictionary{QuotientEdge, Vector{ITensor}}()
    tn = tensornetwork(fn)
    elt = scalartype(tn)
    map_idx = dual_index_map(fn)
    pv = partitioned_vertices
    ket_s = ket_vertex_suffix(fn)
    for pe in quotientedges(ptn)
        src_orig = unique(first.(filter(v -> last(v) == ket_s, pv[parent(src(pe))])))
        dst_orig = unique(first.(filter(v -> last(v) == ket_s, pv[parent(dst(pe))])))
        for (from_orig, to_orig, e) in (
                (src_orig, dst_orig, pe),
                (dst_orig, src_orig, reverse(pe)),
            )
            ms = ITensor[]
            for v_from in from_orig, v_to in to_orig
                for k in commoninds(tn[ket_vertex(fn, v_from)], tn[ket_vertex(fn, v_to)])
                    push!(ms, delta(elt, dag(map_idx(k)), k))
                end
            end
            set!(messages, e, ms)
        end
    end
    return messages
end

function update(qf::QuadraticFormNetwork, original_state_vertex, ket_state::ITensor)
    state_inds = inds(ket_state)
    bra_state = replaceinds(dag(ket_state), state_inds, dual_index_map(qf).(state_inds))
    new_blf = update(
        bilinear_formnetwork(qf),
        original_state_vertex,
        original_state_vertex,
        bra_state,
        ket_state
    )
    return QuadraticFormNetwork(new_blf, dual_index_map(qf), dual_index_map(qf))
end
