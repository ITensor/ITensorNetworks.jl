default_form_type = "Bilinear"

function Form(
  bra::ITensorNetwork,
  operator::ITensorNetwork,
  ket::ITensorNetwork;
  dual_space_map=default_index_map,
  bra_vertex_map=default_bra_vertex_map,
  ket_vertex_map=default_ket_vertex_map,
  operator_vertex_map=default_operator_vertex_map,
  form_type=default_form_type,
)
  dual_map = map(dual_space_map, Indices(externalinds(bra)))

  @assert Set(externalinds(bra)) == Set(externalinds(ket))
  @assert Set(externalinds(operator)) ==
    union(Set(keys(dual_map)), Set(collect(values(dual_map))))
  @assert isempty(findall(in(internalinds(bra)), internalinds(ket)))
  @assert isempty(findall(in(internalinds(bra)), internalinds(operator)))
  @assert isempty(findall(in(internalinds(ket)), internalinds(operator)))

  bra_vs_map = map(bra_vertex_map, Indices(vertices(bra)))
  ket_vs_map = map(ket_vertex_map, Indices(vertices(ket)))
  operator_vs_map = map(operator_vertex_map, Indices(vertices(operator)))

  bra_renamed = rename_vertices_itn(bra, bra_vs_map)
  ket_renamed = rename_vertices_itn(ket, ket_vs_map)
  operator_renamed = rename_vertices_itn(operator, operator_vs_map)
  ket_renamed_dual = map_inds(dual_space_map, ket_renamed; links=[])

  tn = union(union(bra_renamed, operator_renamed), ket_renamed_dual)

  if form_type == "Bilinear"
    return BilinearForm(tn, bra_vs_map, ket_vs_map, operator_vs_map, dual_map)
  elseif form_type == "Quadratic"
    return QuadraticForm(tn, bra_vs_map, ket_vs_map, operator_vs_map, dual_map)
  else
    return error("Form type not supported")
  end
end
