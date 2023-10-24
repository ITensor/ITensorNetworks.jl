default_inds_map(x; kwargs...) = mapprime(x, 0 => 1; kwargs...)
default_inv_inds_map(x; kwargs...) = mapprime(x, 1 => 0; kwargs...)
default_contract_alg(x) = "bp"

# Rayleigh quotient numerator network, ⟨x|A|x⟩
# Also known as a [quadratic form](https://en.wikipedia.org/wiki/Quadratic_form).
# TODO: Allow customizing vertex map.
function quadratic_form_network(
  A::AbstractITensorNetwork,
  x::AbstractITensorNetwork;
  inds_map=default_inds_map,
)
  xAx = ⊗(x, A, inds_map(dag(x)))
  return xAx
end

function quadratic_form_network(x::AbstractITensorNetwork; inds_map=default_inds_map)
  xx = quadratic_form_network(id_network(x; inds_map), x; inds_map)
  return xx
end

function id_network(inds::AbstractIndsNetwork; inds_map=default_inds_map)
  id_net = ITensorNetwork(vertices(inds))
  for v in vertices(inds)
    setindex_preserve_graph!(id_net, ITensor(true), v)
    for i in inds[v]
      setindex_preserve_graph!(id_net, id_net[v] * δ(dag(i), i'), v)
    end
  end
  return id_net
end

function id_network(x::AbstractITensorNetwork; inds_map=default_inds_map)
  return id_network(siteinds(x))
end

# Rayleigh quotient numerator cache, ⟨x|A|x⟩
# Also known as a [quadratic form](https://en.wikipedia.org/wiki/Quadratic_form).
function quadratic_form_cache(
  A::AbstractITensorNetwork,
  x::AbstractITensorNetwork;
  inds_map=default_inds_map,
  inv_inds_map=default_inv_inds_map,
  contract_alg=default_contract_alg(x),
)
  vs = vertices(x)
  xAx = quadratic_form_network(A, x; inds_map)
  xAx_in_vs(v) = (v, 1)
  xAx_out_vs(v) = (v, 3)
  xAx_inds_map = inds_map ∘ dag
  xAx_cache = cache(xAx, vs, xAx_in_vs, xAx_out_vs, xAx_inds_map; contract_alg)
  return xAx_cache
end

# Rayleigh quotient denominator cache, ⟨x|x⟩
# https://en.wikipedia.org/wiki/Norm_(mathematics)
# https://en.wikipedia.org/wiki/Inner_product_space
function quadratic_form_cache(
  x::AbstractITensorNetwork;
  inds_map=default_inds_map,
  kwargs...,
)
  return quadratic_form_cache(id_network(x; inds_map), x; inds_map, kwargs...)
end
