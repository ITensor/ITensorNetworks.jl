using Graphs: vertices
using ITensors: ITensor
using ITensors.NDTensors: Algorithm, @Algorithm_str
using NamedGraphs.Keys: Key
using NamedGraphs.OrdinalIndexing: th

function contraction_sequence(tn::Vector{ITensor}; alg="optimal", kwargs...)
  return contraction_sequence(Algorithm(alg), tn; kwargs...)
end

function contraction_sequence(alg::Algorithm, tn::Vector{ITensor})
  return throw(
    ArgumentError(
      "Algorithm $alg isn't defined for contraction sequence finding. Try loading a backend package like TensorOperations.jl.",
    ),
  )
end

function deepmap(f, tree; filter=(x -> x isa AbstractArray))
  return filter(tree) ? map(t -> deepmap(f, t; filter=filter), tree) : f(tree)
end

function contraction_sequence(tn::AbstractITensorNetwork; kwargs...)
  # TODO: Use `token_vertex` and/or `token_vertices` here.
  ts = map(v -> tn[v], (1:nv(tn))th)
  seq_linear_index = contraction_sequence(ts; kwargs...)
  # TODO: Use `Functors.fmap` or `StructWalk`?
  return deepmap(n -> Key(vertices(tn)[n * th]), seq_linear_index)
end
