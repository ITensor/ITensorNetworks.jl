using NamedGraphs: vertex_to_parent_vertex
using ITensors: ITensor
using ITensors.ContractionSequenceOptimization: deepmap
using ITensors.NDTensors: NDTensors, Algorithm, @Algorithm_str, contract
using LinearAlgebra: normalize!

function NDTensors.contract(tn::AbstractITensorNetwork; alg::String="exact", kwargs...)
  return contract(Algorithm(alg), tn; kwargs...)
end

function NDTensors.contract(
  alg::Algorithm"exact", tn::AbstractITensorNetwork; sequence=vertices(tn), kwargs...
)
  sequence_linear_index = deepmap(v -> vertex_to_parent_vertex(tn, v), sequence)
  return contract(Vector{ITensor}(tn); sequence=sequence_linear_index, kwargs...)
end

function NDTensors.contract(
  alg::Union{Algorithm"density_matrix",Algorithm"ttn_svd"},
  tn::AbstractITensorNetwork;
  output_structure::Function=path_graph_structure,
  kwargs...,
)
  return approx_tensornetwork(alg, tn, output_structure; kwargs...)
end

function contract_density_matrix(
  contract_list::Vector{ITensor}; normalize=true, contractor_kwargs...
)
  tn, _ = contract(
    ITensorNetwork(contract_list); alg="density_matrix", contractor_kwargs...
  )
  out = Vector{ITensor}(tn)
  if normalize
    out .= normalize!.(copy.(out))
  end
  return out
end
