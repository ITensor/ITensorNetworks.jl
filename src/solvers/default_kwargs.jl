"""
  default_kwargs(f, [obj = Any])

Return the default keyword arguments for the function `f`. These defaults may be
derived from the contents or type of the second arugment `obj`.

## Interface

Given a function `f`, one can optionally set the default keyword arguments for this
function by specializing either of the following two-argument methods:
```
ITensorNetworks.default_kwargs(::typeof(f), prob::AbstractProblem)
ITensorNetworks.default_kwargs(::typeof(f), ::Type{<:AbstractProblem})
```
If one does not require the contents of `prob::Prob` to generate the defaults then it is
recommended to dispatch on `Type{<:Prob}` directly (second method) so the defaults
can be accessed without constructing an instance of a `Prob`.

The return value of `default_kwargs` should be a `NamedTuple`, and will overwrite any
default values set in the function signature.
"""
default_kwargs(f) = default_kwargs(f, Any)
default_kwargs(f, obj) = _default_kwargs_fallback(f, obj)

# To avoid annoying potential method ambiguities.
function _default_kwargs_fallback(f, iter::RegionIterator)
  return default_kwargs(f, problem(iter))
end
function _default_kwargs_fallback(f, problem::AbstractProblem)
  return default_kwargs(f, typeof(problem))
end

# Eventually we reach this if nothing is specialized.
_default_kwargs_fallback(::Any, ::DataType) = (;)

"""
  current_kwargs(f, iter::RegionIterator)

Return the keyword arguments to be passed to the function `f` for the current region
defined by the stateful iterator `iter`.
"""
function current_kwargs(f::Function, iter::RegionIterator)
  region_kwargs = get(current_region_kwargs(iter), Symbol(f, :_kwargs), (;))
  rv = merge(default_kwargs(f, iter), region_kwargs)
  return rv
end

# Generic

# I think these should be set independent of a function, but for now:
function default_kwargs(::typeof(factorize), ::Any)
  return (; maxdim=typemax(Int), cutoff=0.0, mindim=1)
end
