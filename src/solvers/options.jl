defaults() = defaults(Any)
defaults(::Any) = (;)

user_defaults() = user_defaults(Any)
user_defaults(::Any) = (;)

# The default case is no defaults exposed at all; they are hardcoded as keyword arguments
# in the function.
#
# defaults(::AbstractProblem, ::Function), in reality, but dont type so one can specialize.
defaults(::Any, ::Any) = (;)
user_defaults(::Any, ::Any) = (;)

"""
Use this function to get options (keyword arguments) from the `RegionIterator` object.
For now we ignore the possibilty of having option packs NOT tied to functions.
"""
function getoption(region_iter::RegionIterator, name=nothing)
  # Get the current specific options for the region
  opt = current_region_kwargs(region_iter)
  prob = problem(region_iter) # We use this to dispatch different defaults

  if isnothing(name)
    # If no `name` then just return the "global" defaults overridden by whatever is in `opt`
    # as a NamedTuple
    return merge(defaults(prob), user_defaults(prob), opt)
  elseif name isa Symbol
    # If `name isa Symbol`, then this refers to a specific global option, so expand global
    # defaults (with overwrites from `opt`) and return this field.
    return getfield(getoption(region_iter), name)
  elseif name isa Function
    # If `name` is a Function, then this refers to a set of options tied to the function
    # `name`, we should expand these defaults, override with the `opt.name` and then return
    # the NamedTuple that results.

    default_opt = defaults(prob, name)
    user_default_opt = user_defaults(prob, name)
    region_opt = get(opt, Symbol(name), (;))

    return merge(default_opt, user_default_opt, region_opt)
  end
end

function expand_defaults(f, region_iter::RegionIterator)
  opt = current_region_kwargs(region_iter)
  prob = problem(region_iter)

  return merge(default_kwargs(f, prob), get(opt, Symbol(name), (;)))
end

function getoption(region_iter::RegionIterator, func::Function, name::Symbol)
  # Returning a specific option of a the options of `func`.
  return getfield(getoption(region_iter, func), name)
end

#=

# Example:

struct MyProblem <: AbstractProblem end

# We have to set the "global" defaults, (if we want to use any), as there is no notion
# of a function where they can be set. If they are in the region plan then that will be used,
# but without defaults set you would have to always have `verbosity` (say) in the region plan
defaults(::MyProblem) = (; verbosity=0)

function compute!(iter::RegionIterator{MyProblem})
  # By default, `getoption` will just splat whatever the region plan opts are!
  extract!(iter; getoption(iter, extract!)...)
  error = update!(iter; getoption(iter, update!)...)

  # This _will_ error if `verbosity` is not defined by `defaults`.
  if getoption(iter, :verbosity) > 0
    @info "Error: $error"
  end

  return iter
end

# Now lets customize the `update!` function for our specific type. Let suppose we are 
# just quickly prototyping and do not care about sharing code and setting defaults etc, we
# can still just use normal keyword arguments.
#
# The return value of `defaults(problem, update!)` overwrites these hard-coded values, but
# by default `defaults(::AbstractProblem, ::Function) = (;)` so overwrites nothing.
function update!(iter::RegionIterator{MyProblem}; maxiter=100, normalize=true)
  total_error = 0

  for _ in 1:maxiter
    state, error = truncation(iter; getoption(iter, truncate)...)
    total_error += error
    if normalize
      state = state / norm(state)
    end
  end

  return total_error
end

# e.g. ...
truncation(iter; kwargs...) = rand(2, 2), 1
extract!(iter; kwargs...) = nothing

# If you now want to share these defaults, then you should define the following:
function defaults(::MyProblem, ::typeof(update!))
  # These will overwrite the keyword defaults. You may want to remove the keyword defaults
  # to remove any ambiguity i.e. `function update!(...; maxiter, norm) ...`
  return (; maxiter=200, normalize=true)
end

# If you want a user to be able to override these defaults with their own defaults (without
# introducing an abstract type) we need another function (this would be set by the user.)
function user_defaults(::MyProblem, ::typeof(update!))
  # This only overwrites the specified default.
  return (; normalize=false)
end

# So, in order of priority, the options get chosen like
# - whatever the options from the region plan are
# - whatever is in `user_defaults`
# - whatever is in `defaults`
# - whatever the keyword argument is set to (if anything).
#
# The `NamedTuple`s in the region plan only need to have one layer of nesting, i.e. the
# "global options" (if any) and the function option packs.

function test()
  ri = RegionIterator(
    MyProblem(), ["region" => ((update!)=(; maxiter=300), verbosity=1)], 1
  )
  compute!(ri)
  return nothing
end

=#
