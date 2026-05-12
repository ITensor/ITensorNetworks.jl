using Adapt: adapt
using DataGraphs: DataGraphs, set_vertex_data!
using ITensors.NDTensors: datatype, denseblocks
using ITensors: ITensor, Op, delta, prime, sim
using NamedGraphs.GraphsExtensions: disjoint_union

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
