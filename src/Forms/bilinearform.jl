default_index_map = prime
default_bra_vertex_map(v) = (v, "bra")
default_ket_vertex_map(v) = (v, "ket")
default_operator_vertex_map(v) = (v, "operator")
default_tno_constructor(s::IndsNetwork) = delta_network(s)

struct BilinearForm{V,KetMap,BraMap,OperatorMap,SpaceMap} <:
       AbstractForm{V,KetMap,BraMap,OperatorMap,SpaceMap}
  tn::ITensorNetwork{V}
  bra_vs_map::BraMap
  ket_vs_map::KetMap
  operator_vs_map::OperatorMap
  dual_space_map::SpaceMap
end

#Needed for implementation
bra_map(blf::BilinearForm) = blf.bra_vs_map
ket_map(blf::BilinearForm) = blf.ket_vs_map
operator_map(blf::BilinearForm) = blf.operator_vs_map
space_map(blf::BilinearForm) = blf.dual_space_map
tensornetwork(blf::BilinearForm) = blf.tn
data_graph_type(::Type{<:BilinearForm}) = data_graph_type(tensornetwork(blf))
data_graph(blf::BilinearForm) = data_graph(tensornetwork(blf))

function BilinearForm(
  bra::ITensorNetwork, operator::ITensorNetwork, ket::ITensorNetwork; kwargs...
)
  return Form(bra, operator, ket; form_type="Bilinear", kwargs...)
end

function BilinearForm(
  bra::ITensorNetwork,
  ket::ITensorNetwork;
  dual_space_map=default_index_map,
  tno_constructor=default_tno_constructor,
  kwargs...,
)
  s = siteinds(bra)
  operator_space = union_all_inds(s, dual_space_map(s; links=[]))
  O = tno_constructor(operator_space)
  return BilinearForm(bra, O, ket; kwargs...)
end
