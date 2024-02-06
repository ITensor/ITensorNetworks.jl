module ITensorNetworksEinExprsExt

using ITensors
using ITensorNetworks
using ITensorNetworks: Index
using EinExprs: EinExprs, EinExpr, einexpr, SizedEinExpr

function EinExprs.einexpr(tn::ITensorNetwork{T}; optimizer::EinExprs.Optimizer) where {T}
  IndexType = Any

  tensors = EinExpr{IndexType}[]
  tensor_map = Dict{Set{IndexType},T}()
  sizedict = Dict{IndexType,Int}()

  for (key, tensor) in pairs(vertex_data(tn))
    _inds = collect(ITensorNetworks.inds(tensor))
    push!(tensors, EinExpr{IndexType}(; head=_inds))
    tensor_map[Set(_inds)] = key
    merge!(sizedict, Dict(_inds .=> size(tensor)))
  end

  _openinds = collect(externalinds(tn))
  expr = SizedEinExpr(sum(tensors; skip=_openinds), sizedict)

  return einexpr(optimizer, expr)
end

function ITensorNetworks.contraction_sequence(
  ::ITensorNetworks.Algorithm"einexpr", tn::Vector{ITensor}; optimizer=EinExprs.Exhaustive()
)
  tn = ITensorNetwork(tn)
  IndexType = Any

  tensors = EinExpr{IndexType}[]
  tensor_map = Dict{Set{IndexType},T}()
  sizedict = Dict{IndexType,Int}()

  for (key, tensor) in pairs(vertex_data(tn))
    _inds = collect(ITensorNetworks.inds(tensor))
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
