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

function EinExprs.einexpr(tn::ITensorNetwork; optimizer::EinExprs.Optimizer)
  IndexType = Any
  VertexType = vertextype(tn)

  tensors = EinExpr{IndexType}[]
  tensor_map = Dict{Set{IndexType},VertexType}()
  sizedict = Dict{IndexType,Int}()

  for (key, tensor) in pairs(vertex_data(tn))
    _inds = collect(inds(tensor))
    push!(tensors, EinExpr{IndexType}(; head=_inds))
    tensor_map[Set(_inds)] = key
    merge!(sizedict, Dict(_inds .=> size(tensor)))
  end

  _openinds = collect(externalinds(tn))
  expr = SizedEinExpr(sum(tensors; skip=_openinds), sizedict)

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
  IndexType = Any
  VertexType = vertextype(tn)

  tensors = EinExpr{IndexType}[]
  tensor_map = Dict{Set{IndexType},VertexType}()
  sizedict = Dict{IndexType,Int}()

  for (key, tensor) in pairs(vertex_data(tn))
    _inds = collect(inds(tensor))
    push!(tensors, EinExpr{IndexType}(; head=_inds))
    tensor_map[Set(_inds)] = key
    merge!(sizedict, Dict(_inds .=> size(tensor)))
  end

  _openinds = collect(externalinds(tn))
  expr = SizedEinExpr(sum(tensors; skip=_openinds), sizedict)

  path = einexpr(optimizer, expr)

  function _convert_to_contraction_sequence(subpath)
    EinExprs.nargs(subpath) == 0 && return tensor_map[Set(subpath.head)]
    return map(_convert_to_contraction_sequence, EinExprs.args(subpath))
  end

  return _convert_to_contraction_sequence(path)
end

end
