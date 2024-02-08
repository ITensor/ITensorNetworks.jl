struct QuadraticForm{V,KetMap,BraMap,OperatorMap,VectorSpaceMap,LinkSpaceMap} <:
       AbstractForm{V,KetMap,BraMap,OperatorMap,VectorSpaceMap}
  tn::ITensorNetwork{V}
  bra_vs_map::BraMap
  ket_vs_map::KetMap
  operator_vs_map::OperatorMap
  dual_space_map::VectorSpaceMap
  link_space_map::LinkSpaceMap
end

#Needed for implementation
bra_map(qf::QuadraticForm) = qf.bra_vs_map
ket_map(qf::QuadraticForm) = qf.ket_vs_map
operator_map(qf::QuadraticForm) = qf.operator_vs_map
space_map(qf::QuadraticForm) = qf.dual_space_map
tensornetwork(qf::QuadraticForm) = qf.tn
data_graph_type(::Type{<:QuadraticForm}) = data_graph_type(tensornetwork(qf))
data_graph(qf::QuadraticForm) = data_graph(tensornetwork(qf))

function QuadraticForm(
  ket::ITensorNetwork, operator::ITensorNetwork; link_space_map=default_index_map, kwargs...
)
  bra = link_space_map(dag(ket); sites=[])
  return Form(bra, operator, ket; form_type="Quadratic", link_space_map, kwargs...)
end

function QuadraticForm(
  ket::ITensorNetwork;
  dual_space_map=default_index_map,
  tno_constructor=default_tno_constructor,
  kwargs...,
)
  s = siteinds(ket)
  operator_space = union_all_inds(s, dual_space_map(s; links=[]))
  operator = tno_constructor(operator_space)
  return QuadraticForm(ket, operator; kwargs...)
end

function bra_ket_vertices(qf::QuadraticForm, state_vertices::Vector)
  ket_vertices = [ket_map(qf)[sv] for sv in state_vertices]
  bra_vertices = [bra_map(qf)[sv] for sv in state_vertices]
  return bra_vertices, ket_vertices
end

function gradient(
  qf::QuadraticForm, state_vertices::Vector; alg="Exact", (cache!)=nothing, kwargs...
)
  return gradient(Algorithm(alg), qf, state_vertices; (cache!), kwargs...)
end

function gradient(
  ::Algorithm"Exact",
  qf::QuadraticForm,
  state_vertices::Vector;
  (cache!)=nothing,
  contractor_kwargs...,
)
  qf_bra_vertices, qf_ket_vertices = bra_ket_vertices(qf, state_vertices)
  tn_vertices = setdiff(vertices(qf), vcat(qf_bra_vertices, qf_ket_vertices))
  reduced_tn, _ = induced_subgraph(tensornetwork(qf), tn_vertices)
  return contract(reduced_tn, contractor_kwargs...)
end

function update_qf!(qf::QuadraticForm, state_vertex, state::ITensor)
  setindex_preserve_graph!(tensornetwork(qf)[ket_map(state_vertex)], state, state_vertex)
  #Change to be consistent with dual space map and link space map. How can we do this elegantly??
  return setindex_preserve_graph!(
    tensornetwork(qf)[bra_map(state_vertex)], prime(dag(state)), state_vertex
  )
end
