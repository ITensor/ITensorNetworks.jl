using DataGraphs: DataGraphs, set_vertex_data!, underlying_graph, vertex_data
using Dictionaries: Dictionary
using Graphs: is_tree
using ITensors.NDTensors: @Algorithm_str, Algorithm
using NamedGraphs.PartitionedGraphs: PartitionedGraph, quotient_graph

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

function identity_messages(fn::QuadraticFormNetwork, ptn::PartitionedGraph)
    return identity_messages(bilinear_formnetwork(fn), ptn)
end

# QFN is structurally ψ-vs-ψ, so identity messages are the canonical
# initialization on a loopy quotient graph. Asymmetric form networks
# (general LFN/BFN built from ϕ ≠ ψ) don't have this guarantee, so this
# specialization is QFN-only — `inner(ϕ, ψ; alg = "bp")` still falls
# back to the generic path and requires the caller to supply messages.
function logscalar(
        alg::Algorithm"bp",
        fn::QuadraticFormNetwork;
        (cache!) = nothing,
        update_cache = isnothing(cache!),
        cache_update_kwargs = (;)
    )
    if isnothing(cache!)
        pv = default_partitioned_vertices(fn)
        ptn = PartitionedGraph(fn, pv)
        messages = is_tree(quotient_graph(ptn)) ? Dictionary() : identity_messages(fn, ptn)
        cache! = Ref(BeliefPropagationCache(ptn; messages))
    end
    if update_cache
        cache![] = update(cache![]; cache_update_kwargs...)
    end
    return logscalar(cache![])
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
