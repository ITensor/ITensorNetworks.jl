using Graphs: induced_subgraph
using NamedGraphs.SimilarType: SimilarType

default_bra_vertex_suffix() = "bra"
default_ket_vertex_suffix() = "ket"
default_operator_vertex_suffix() = "operator"

abstract type AbstractFormNetwork{V} <: AbstractITensorNetwork{V} end

#Needed for interface
dual_index_map(f::AbstractFormNetwork) = not_implemented()
# TODO: Use `NamedGraphs.GraphsExtensions.parent_graph`.
tensornetwork(f::AbstractFormNetwork) = not_implemented()
Base.copy(f::AbstractFormNetwork) = not_implemented()
operator_vertex_suffix(f::AbstractFormNetwork) = not_implemented()
bra_vertex_suffix(f::AbstractFormNetwork) = not_implemented()
ket_vertex_suffix(f::AbstractFormNetwork) = not_implemented()

function SimilarType.similar_type(f::AbstractFormNetwork)
    return typeof(tensornetwork(f))
end

# TODO: Use `NamedGraphs.GraphsExtensions.parent_graph_type`.
function data_graph_type(f::AbstractFormNetwork)
    return data_graph_type(tensornetwork(f))
end
# TODO: Use `NamedGraphs.GraphsExtensions.parent_graph`.
data_graph(f::AbstractFormNetwork) = data_graph(tensornetwork(f))

function operator_vertices(f::AbstractFormNetwork)
    return filter(v -> last(v) == operator_vertex_suffix(f), vertices(f))
end

function bra_vertices(f::AbstractFormNetwork)
    return filter(v -> last(v) == bra_vertex_suffix(f), vertices(f))
end

function ket_vertices(f::AbstractFormNetwork)
    return filter(v -> last(v) == ket_vertex_suffix(f), vertices(f))
end

function operator_vertices(f::AbstractFormNetwork, original_state_vertices::Vector)
    return [operator_vertex_map(f)(osv) for osv in original_state_vertices]
end

function bra_vertices(f::AbstractFormNetwork, original_state_vertices::Vector)
    return [bra_vertex_map(f)(osv) for osv in original_state_vertices]
end

function ket_vertices(f::AbstractFormNetwork, original_state_vertices::Vector)
    return [ket_vertex_map(f)(osv) for osv in original_state_vertices]
end

function state_vertices(f::AbstractFormNetwork)
    return vcat(bra_vertices(f), ket_vertices(f))
end

function state_vertices(f::AbstractFormNetwork, original_state_vertices::Vector)
    return vcat(
        bra_vertices(f, original_state_vertices), ket_vertices(f, original_state_vertices)
    )
end

function Graphs.induced_subgraph(f::AbstractFormNetwork, vertices::Vector)
    return induced_subgraph(tensornetwork(f), vertices)
end

function bra_network(f::AbstractFormNetwork)
    return rename_vertices(inv_vertex_map(f), first(induced_subgraph(f, bra_vertices(f))))
end

function ket_network(f::AbstractFormNetwork)
    return rename_vertices(inv_vertex_map(f), first(induced_subgraph(f, ket_vertices(f))))
end

function operator_network(f::AbstractFormNetwork)
    return rename_vertices(
        inv_vertex_map(f), first(induced_subgraph(f, operator_vertices(f)))
    )
end

operator_vertex_map(f::AbstractFormNetwork) = v -> (v, operator_vertex_suffix(f))
bra_vertex_map(f::AbstractFormNetwork) = v -> (v, bra_vertex_suffix(f))
ket_vertex_map(f::AbstractFormNetwork) = v -> (v, ket_vertex_suffix(f))
inv_vertex_map(f::AbstractFormNetwork) = v -> first(v)
operator_vertex(f::AbstractFormNetwork, v) = operator_vertex_map(f)(v)
bra_vertex(f::AbstractFormNetwork, v) = bra_vertex_map(f)(v)
ket_vertex(f::AbstractFormNetwork, v) = ket_vertex_map(f)(v)
original_state_vertex(f::AbstractFormNetwork, v) = inv_vertex_map(f)(v)

function default_partitioned_vertices(f::AbstractFormNetwork)
    return group(v -> original_state_vertex(f, v), vertices(f))
end
