using Graphs: vertices
using ITensorBase: ITensor
## using ITensors.NDTensors: Algorithm, @Algorithm_str
using NamedGraphs.Keys: Key
using NamedGraphs.OrdinalIndexing: th

const ITensorList = Union{Vector{ITensor},Tuple{Vararg{ITensor}}}

function contraction_sequence(tn::ITensorList; alg="optimal", kwargs...)
  return contraction_sequence(Algorithm(alg), tn; kwargs...)
end

function contraction_sequence(alg::Algorithm, tn::ITensorList)
  return throw(
    ArgumentError(
      "Algorithm $alg isn't defined for contraction sequence finding. Try loading a backend package like 
        TensorOperations.jl or OMEinsumContractionOrders.jl.",
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
