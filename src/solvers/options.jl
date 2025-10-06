defaults(::Any, ::Any) = (;)

"""
  function default_kwargs(f, iter::RegionIterator)

Return the default keyword arguments for the function `f` overridden by the contents of `iter`.
"""
function default_kwargs(f::Function, iter::RegionIterator)
  region_kwargs = get(current_region_kwargs(iter), Symbol(f, :_kwargs), (;))
  return merge(default_kwargs(f, problem(iter)), region_kwargs)
end
