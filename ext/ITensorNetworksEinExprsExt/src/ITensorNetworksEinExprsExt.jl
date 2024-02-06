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
  tensor_map = Dict{Set{IndexType},VertexType}()
  sizedict = Dict{IndexType,Int}()

  for v in vertices(tn)
    tensor_v = tn[v]
    inds_v = collect(inds(tensor_v))
    push!(tensor_exprs, EinExpr{IndexType}(; head=inds_v))
    tensor_map[Set(inds_v)] = key
    merge!(sizedict, Dict(inds_v .=> size(tensor_v)))
  end

  externalinds_tn = collect(externalinds(tn))
  expr = SizedEinExpr(sum(tensor_exprs; skip=externalinds_tn), sizedict)

  return expr, tensor_map
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
  path, tensor_map = prepare_einexpr(tn)
  return to_contraction_sequence(tensor_map, path)
end

function to_contraction_sequence(tensor_map, path)
  EinExprs.nargs(path) == 0 && return tensor_map[Set(path.head)]
  return map(_convert_to_contraction_sequence, EinExprs.args(path))
end

end
