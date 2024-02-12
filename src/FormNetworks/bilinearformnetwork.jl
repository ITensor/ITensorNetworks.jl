default_bra_vertex_map(v) = (v, "bra")
default_ket_vertex_map(v) = (v, "ket")
default_operator_vertex_map(v) = (v, "operator")
default_operator_constructor(s::IndsNetwork) = delta_network(s)

struct BilinearFormNetwork{V,TensorNetwork<:AbstractITensorNetwork{V}
,BraMap,KetMap,OperatorMap} <: AbstractFormNetwork{V}
  tensornetwork::TensorNetwork
  bra_vertex_map::BraMap
  ket_vertex_map::KetMap
  operator_vertex_map::OperatorMap
end

function BilinearFormNetwork(
  operator::AbstractITensorNetwork,
  bra::AbstractITensorNetwork,
  ket::AbstractITensorNetwork;
  bra_vertex_map=default_bra_vertex_map,
  ket_vertex_map=default_ket_vertex_map,
  operator_vertex_map=default_operator_vertex_map,
)
  @assert Set(externalinds(operator)) ==
    union(Set(externalinds(ket)), Set(externalinds(bra)))
  @assert isempty(findall(in(internalinds(bra)), internalinds(ket)))
  @assert isempty(findall(in(internalinds(bra)), internalinds(operator)))
  @assert isempty(findall(in(internalinds(ket)), internalinds(operator)))

  bra_renamed = rename_vertices_itn(bra, bra_vertex_map)
  ket_renamed = rename_vertices_itn(ket, ket_vertex_map)
  operator_renamed = rename_vertices_itn(operator, operator_vertex_map)

  # TODO: Reminder to fix `union` so that `union(bra_renamed, operator_renamed, ket_renamed)` works.
  tn = union(union(bra_renamed, operator_renamed), ket_renamed)

  return BilinearFormNetwork(tn, bra_vertex_map, ket_vertex_map, operator_vertex_map)
end

#Needed for implementation
bra_vertex_map(blf::BilinearFormNetwork) = blf.bra_vertex_map
ket_vertex_map(blf::BilinearFormNetwork) = blf.ket_vertex_map
operator_vertex_map(blf::BilinearFormNetwork) = blf.operator_vertex_map
tensornetwork(blf::BilinearFormNetwork) = blf.tn
data_graph_type(::Type{<:BilinearFormNetwork}) = data_graph_type(tensornetwork(blf))
data_graph(blf::BilinearFormNetwork) = data_graph(tensornetwork(blf))

function copy(blf::BilinearFormNetwork)
  return BilinearFormNetwork(
    copy(tensornetwork(blf)),
    bra_vertex_map(blf),
    ket_vertex_map(blf),
    operator_vertex_map(blf),
  )
end

function BilinearFormNetwork(
  bra::AbstractITensorNetwork,
  ket::AbstractITensorNetwork;
  operator_constructor=default_operator_constructor,
  kwargs...,
)
  operator_space = union_all_inds(siteinds(bra), siteinds(ket))
  O = tno_constructor(operator_space)
  return BilinearFormNetwork(bra, O, ket; kwargs...)
end
