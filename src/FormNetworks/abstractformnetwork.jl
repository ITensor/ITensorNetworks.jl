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

bra(f::AbstractFormNetwork) = induced_subgraph(f, collect(values(bra_vertex_map(f))))
ket(f::AbstractFormNetwork) = induced_subgraph(f, collect(values(ket_vertex_map(f))))
function operator(f::AbstractFormNetwork)
  return induced_subgraph(f, collect(values(operator_vertex_map(f))))
end

function derivative(f::AbstractFormNetwork, state_vertices::Vector; kwargs...)
  tn_vertices = derivative_vertices(f, state_vertices)
  return derivative(tensornetwork(f), tn_vertices; kwargs...)
end

function bra_vertices(f::AbstractFormNetwork, state_vertices::Vector)
  return [bra_vertex_map(f)(sv) for sv in state_vertices]
end

function ket_vertices(f::AbstractFormNetwork, state_vertices::Vector)
  return [ket_vertex_map(f)(sv) for sv in state_vertices]
end

function derivative_vertices(f::AbstractFormNetwork, state_vertices::Vector; kwargs...)
  return setdiff(
    vertices(f), vcat(bra_vertices(f, state_vertices), ket_vertices(f, state_vertices))
  )
end

operator_vertex_map(f::AbstractFormNetwork) = v -> (v, operator_vertex_suffix(f))
bra_vertex_map(f::AbstractFormNetwork) = v -> (v, bra_vertex_suffix(f))
ket_vertex_map(f::AbstractFormNetwork) = v -> (v, ket_vertex_suffix(f))
