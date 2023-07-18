function contract(tn::AbstractITensorNetwork; alg::String="exact", kwargs...)
  return contract(Algorithm(alg), tn; kwargs...)
end

function contract(
  alg::Algorithm"exact", tn::AbstractITensorNetwork; sequence=vertices(tn), kwargs...
)
  sequence_linear_index = deepmap(v -> vertex_to_parent_vertex(tn, v), sequence)
  return contract(Vector{ITensor}(tn); sequence=sequence_linear_index, kwargs...)
end

function contract(
  alg::Union{Algorithm"density_matrix",Algorithm"density_matrix_direct_eigen",Algorithm"ttn_svd"},
  tn::AbstractITensorNetwork;
  output_structure::Function=path_graph_structure,
  kwargs...,
)
  return approx_itensornetwork(alg, tn, output_structure; kwargs...)
end

function contract(
  alg::Algorithm"partitioned_contract",
  tn::AbstractITensorNetwork;
  nvertices_per_partition::Integer=2,
  backend::String="KaHyPar",
  kwargs...,
)
  sequence = _partitioned_contraction_sequence(
    tn; nvertices_per_partition=nvertices_per_partition, backend=backend
  )
  return partitioned_contract(sequence; kwargs...)
end

function _partitioned_contraction_sequence(
  tn::ITensorNetwork; nvertices_per_partition=2, backend="KaHyPar"
)
  # @assert is_connected(tn)
  g_parts = partition(tn; npartitions=2, backend=backend)
  if nv(g_parts[1]) >= max(nvertices_per_partition, 2)
    tntree_1 = _partitioned_contraction_sequence(
      g_parts[1]; nvertices_per_partition, backend
    )
  else
    tntree_1 = Vector{ITensor}(g_parts[1])
  end
  if nv(g_parts[2]) >= max(nvertices_per_partition, 2)
    tntree_2 = _partitioned_contraction_sequence(
      g_parts[2]; nvertices_per_partition, backend=backend
    )
  else
    tntree_2 = Vector{ITensor}(g_parts[2])
  end
  return [tntree_1, tntree_2]
end
