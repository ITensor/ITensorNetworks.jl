abstract type AbstractFormNetwork{V} <: AbstractITensorNetwork{V} end

#Needed for interface
bra_vertex_map(f::AbstractFormNetwork) = not_implemented()
ket_vertex_map(f::AbstractFormNetwork) = not_implemented()
operator_vertex_map(f::AbstractFormNetwork) = not_implemented()
dual_index_map(f::AbstractFormNetwork) = not_implemented()
tensornetwork(f::AbstractFormNetwork) = not_implemented()
copy(f::AbstractFormNetwork) = not_implemented()
derivative_vertices(f::AbstractFormNetwork) = not_implemented()

bra(f::AbstractFormNetwork) = induced_subgraph(f, collect(values(bra_vertex_map(f))))
ket(f::AbstractFormNetwork) = induced_subgraph(f, collect(values(ket_vertex_map(f))))
function operator(f::AbstractFormNetwork)
  return induced_subgraph(f, collect(values(operator_vertex_map(f))))
end

function derivative(f::AbstractFormNetwork, state_vertices::Vector; kwargs...)
  tn_vertices = derivative_vertices(f, state_vertices)
  return derivative(tensornetwork(f), tn_vertices; kwargs...)
end
