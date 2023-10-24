# ⟨x|A|x⟩ / ⟨x|x⟩
struct RayleighQuotientCache{Num<:AbstractITNCache,Den<:AbstractITNCache} <: AbstractITNCache
  num::Num
  den::Den
end

function set_nsite(cache::RayleighQuotientCache, nsite)
  # TODO: Use `set_numerator` and `set_denominator`.
  return RayleighQuotientCache(set_nsite(cache.num, nsite), set_nsite(cache.den, nsite))
end

function (cache::RayleighQuotientCache)(v)
  return cache.num(v)
end

# TODO: Change to `set_update_region`.
function position(cache::RayleighQuotientCache, tns::AbstractITensorNetwork, region)
  # TODO: Use `set_numerator` and `set_denominator`.
  return RayleighQuotientCache(position(cache.num, tns, region), position(cache.den, tns, region))
end

# Cache for a tensor network representation of a
# [Rayleigh quotient](https://en.wikipedia.org/wiki/Rayleigh_quotient).
# TODO: Detect if there is an orthogonality center and if so
# avoid making the denominator cache of the Rayleigh quotient.
function rayleigh_quotient_cache(
  A::AbstractITensorNetwork,
  x::AbstractITensorNetwork;
  kwargs...,
)
  xAx_cache = quadratic_form_cache(A, x; kwargs...)
  xx_cache = quadratic_form_cache(x; kwargs...)
  return RayleighQuotientCache(xAx_cache, xx_cache)
end
