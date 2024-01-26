module ITensorNetworksEinExprsExt

using ITensorNetworks
using EinExprs: EinExprs, EinExpr, einexpr

function ITensorNetworks.contraction_sequence(
  ::ITensorNetworks.Algorithm"einexpr", tn; optimizer=EinExprs.Exhaustive()
)
  tensor_map = IdDict(
    map(pairs(vertex_data(tn))) do (key, tensor)
      _inds = collect(ITensorNetworks.inds(tensor))
      _size = Dict(_inds .=> size(tensor))
      EinExpr(_inds, _size) => key
    end,
  )
  tensors = collect(keys(tensor_map))

  _openinds = collect(ITensorNetworks.externalinds(tn))
  expr = sum(tensors; skip=_openinds)
  path = einexpr(optimizer, expr)

  function _convert_to_contraction_sequence(subpath)
    length(subpath.args) == 0 && return tensor_map[subpath]
    return map(_convert_to_contraction_sequence, subpath.args)
  end

  return _convert_to_contraction_sequence(path)
end

end
