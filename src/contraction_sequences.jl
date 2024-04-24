using Graphs: vertices
using ITensors: ITensor, contract
using ITensors.ContractionSequenceOptimization: deepmap, optimal_contraction_sequence
using ITensors.NDTensors: Algorithm, @Algorithm_str
using NamedGraphs: ordinal_vertex_to_vertex
using NamedGraphs.Keys: Key

function contraction_sequence(tn::Vector{ITensor}; alg="optimal", kwargs...)
  return contraction_sequence(Algorithm(alg), tn; kwargs...)
end

function contraction_sequence(tn::AbstractITensorNetwork; kwargs...)
  # TODO: Use `token_vertex` and/or `token_vertices` here.
  ts = map(pv -> tn[ordinal_vertex_to_vertex(tn, pv)], 1:nv(tn))
  seq_linear_index = contraction_sequence(ts; kwargs...)
  # TODO: Use `Functors.fmap` or `StructWalk`?
  return deepmap(n -> Key(ordinal_vertex_to_vertex(tn, n)), seq_linear_index)
end

function contraction_sequence(::Algorithm"optimal", tn::Vector{ITensor})
  return optimal_contraction_sequence(tn)
end
