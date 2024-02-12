default_bra_vertex_suffix() = "bra"
default_ket_vertex_suffix() = "ket"
default_operator_vertex_suffix() = "operator"

abstract type AbstractFormNetwork{V} <: AbstractITensorNetwork{V} end

#Needed for interface
dual_index_map(f::AbstractFormNetwork) = not_implemented()
tensornetwork(f::AbstractFormNetwork) = not_implemented()
copy(f::AbstractFormNetwork) = not_implemented()
bra_ket_vertices(f::AbstractFormNetwork, state_vertices::Vector) = not_implemented()
operator_vertex_suffix(f::AbstractFormNetwork) = not_implemented()
bra_vertex_suffix(f::AbstractFormNetwork) = not_implemented()
ket_vertex_suffix(f::AbstractFormNetwork) = not_implemented()

function operator_vertices(f::AbstractFormNetwork)
  return filter(v -> last(v) == operator_vertex_suffix(f), vertices(f))
end
function bra_vertices(f::AbstractFormNetwork)
  return filter(v -> last(v) == bra_vertex_suffix(f), vertices(f))
end
function ket_vertices(f::AbstractFormNetwork)
  return filter(v -> last(v) == ket_vertex_suffix(f), vertices(f))
end

function bra_vertices(f::AbstractFormNetwork, state_vertices::Vector)
  return [bra_vertex_map(f)(sv) for sv in state_vertices]
end

function ket_vertices(f::AbstractFormNetwork, state_vertices::Vector)
  return [ket_vertex_map(f)(sv) for sv in state_vertices]
end

function Graphs.induced_subgraph(f::AbstractFormNetwork, vertices::Vector)
  return induced_subgraph(tensornetwork(f), vertices)
end
function bra(f::AbstractFormNetwork)
  return rename_vertices(inv_vertex_map(f), first(induced_subgraph(f, bra_vertices(f))))
end
function ket(f::AbstractFormNetwork)
  return rename_vertices(inv_vertex_map(f), first(induced_subgraph(f, ket_vertices(f))))
end
function operator(f::AbstractFormNetwork)
  return rename_vertices(
    inv_vertex_map(f), first(induced_subgraph(f, operator_vertices(f)))
  )
end

function derivative(f::AbstractFormNetwork, state_vertices::Vector; kwargs...)
  tn_vertices = derivative_vertices(f, state_vertices)
  return derivative(tensornetwork(f), tn_vertices; kwargs...)
end

function derivative_vertices(f::AbstractFormNetwork, state_vertices::Vector; kwargs...)
  return setdiff(
    vertices(f), vcat(bra_vertices(f, state_vertices), ket_vertices(f, state_vertices))
  )
end

operator_vertex_map(f::AbstractFormNetwork) = v -> (v, operator_vertex_suffix(f))
bra_vertex_map(f::AbstractFormNetwork) = v -> (v, bra_vertex_suffix(f))
ket_vertex_map(f::AbstractFormNetwork) = v -> (v, ket_vertex_suffix(f))
inv_vertex_map(f::AbstractFormNetwork) = v -> first(v)
