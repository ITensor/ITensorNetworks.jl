module ITensorNetworksEinExprsExt

using ITensors: Index, ITensor, Algorithm, noncommoninds
using ITensorNetworks:
  ITensorNetworks, ITensorNetwork, vertextype, vertex_data, contraction_sequence
using EinExprs: EinExprs, EinExpr, einexpr, SizedEinExpr

function to_einexpr(ts::Vector{ITensor})
  IndexType = Any

  tensor_exprs = EinExpr{IndexType}[]
  inds_dims = Dict{IndexType,Int}()

  for tensor_v in ts
    inds_v = collect(inds(tensor_v))
    push!(tensor_exprs, EinExpr{IndexType}(; head=inds_v))
    merge!(inds_dims, Dict(inds_v .=> size(tensor_v)))
  end

  externalinds_tn = collect(noncommoninds(ts...))
  return SizedEinExpr(sum(tensor_exprs; skip=externalinds_tn), inds_dims)
end

function tensor_inds_to_vertex(ts::Vector{ITensor})
  mapping = Dict{Set{IndexType},VertexType}()

  for (v, tensor_v) in enumerate(ts)
    inds_v = collect(inds(tensor_v))
    mapping[Set(inds_v)] = v
  end

  return mapping
end

function ITensorNetworks.contraction_sequence(
  ::Algorithm"einexpr", tn::Vector{ITensor}; optimizer=EinExprs.Exhaustive()
)
  expr = to_einexpr(tn)
  path = einexpr(expr; kwargs...)
  return to_contraction_sequence(path, tensor_inds_to_vertex(tn))
end

function to_contraction_sequence(expr, tensor_inds_to_vertex)
  EinExprs.nargs(expr) == 0 && return tensor_inds_to_vertex[Set(expr.head)]
  return map(
    expr -> to_contraction_sequence(expr, tensor_inds_to_vertex), EinExprs.args(expr)
  )
end

end
