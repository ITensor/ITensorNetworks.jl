module ITensorNetworksEinExprsExt

using ITensors: Index, ITensor, Algorithm
using ITensorNetworks:
  ITensorNetworks,
  ITensorNetwork,
  vertextype,
  vertex_data,
  externalinds,
  contraction_sequence
using EinExprs: EinExprs, EinExpr, einexpr, SizedEinExpr

function prepare_einexpr(tn::ITensorNetwork)
  IndexType = Any
  VertexType = vertextype(tn)

  tensor_exprs = EinExpr{IndexType}[]
  tensor_inds_to_vertex = Dict{Set{IndexType},VertexType}()
  inds_dims = Dict{IndexType,Int}()

  for v in vertices(tn)
    tensor_v = tn[v]
    inds_v = collect(inds(tensor_v))
    push!(tensor_exprs, EinExpr{IndexType}(; head=inds_v))
    tensor_inds_to_vertex[Set(inds_v)] = key
    merge!(inds_dims, Dict(inds_v .=> size(tensor_v)))
  end

  externalinds_tn = collect(externalinds(tn))
  expr = SizedEinExpr(sum(tensor_exprs; skip=externalinds_tn), inds_dims)

  return expr, tensor_inds_to_vertex
end

function EinExprs.einexpr(tn::ITensorNetwork; optimizer::EinExprs.Optimizer)
  expr, _ = prepare_einexpr(tn)
  return einexpr(optimizer, expr)
end

function ITensorNetworks.contraction_sequence(
  alg::Algorithm"einexpr", tn::Vector{ITensor}; kwargs...
)
  return contraction_sequence(alg, ITensorNetwork(tn); kwargs...)
end

function ITensorNetworks.contraction_sequence(
  ::Algorithm"einexpr", tn::ITensorNetwork{T}; optimizer=EinExprs.Exhaustive()
)
  expr, tensor_inds_to_vertex = to_einexpr(tn)
  return to_contraction_sequence(tensor_inds_to_vertex, expr)
end

function to_contraction_sequence(tensor_inds_to_vertex, expr)
  EinExprs.nargs(expr) == 0 && return tensor_inds_to_vertex[Set(expr.head)]
  return map(_convert_to_contraction_sequence, EinExprs.args(expr))
end

end
