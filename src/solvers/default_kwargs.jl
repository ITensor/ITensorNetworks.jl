using MacroTools

"""
  default_kwargs(f::Function, args...; kwargs...)

Returns a set of default keyword arguments, as a `NamedTuple`, for the function `f` 
depending on an arbitrary number of positional arguments. Any number of these default
keyword arguments can optionally be overwritten by passing the the keyword as a 
keyword argument to this function.
"""
function default_kwargs(f::Function, args...; kwargs...)
  return default_kwargs(f, map(typeof, args)...; kwargs...)
end
default_kwargs(f::Function, ::Vararg{<:Type}; kwargs...) = (; kwargs...)

"""
    @default_kwargs

Automatically define a `default_kwargs` method for a given function. This macro should
be applied before a function definition:
```
@default_kwargs astypes = true function f(args...; kwargs...) 
  ...
end
```
If `astypes = true` then the `default_kwargs` method is defined in the 
type domain with respect to `args`, i.e.
```
default_kwargs(::typeof(f), arg::T; kwargs...) # astypes = false
default_kwargs(::typeof(f), arg::Type{<:T}; kwargs...) # astypes = true
```
"""
macro default_kwargs(args...)
  kwargs = (;)
  for opt in args
    if @capture(opt, key_ = val_)
      @info "" key val
      kwargs = merge(kwargs, NamedTuple{(key,)}((val,)))
    elseif opt === last(args)
      return default_kwargs_macro(opt; kwargs...)
    else
      throw(ArgumentError("Unknown expression object"))
    end
  end
end

function default_kwargs_macro(function_def; astypes=true)
  if !isdef(function_def)
    throw(
      ArgumentError("The @default_kwargs macro must be followed by a function definition")
    )
  end

  ex = splitdef(function_def)
  new_ex = deepcopy(ex)

  prev_kwargs = []

  # Give very positional argument a name and escape the type.
  ex[:args] = map(ex[:args]) do arg
    @capture(arg, (name_::T_) | (::T_) | name_)
    if isnothing(name)
      name = gensym()
    end
    if isnothing(T)
      T = :Any
    end
    return :($(name)::$(esc(T)))
  end

  # Replacing the kwargs values with the output of `default_kwargs`
  ex[:kwargs] = map(ex[:kwargs]) do kw
    @capture(kw, (key_::T_ = val_) | (key_ = val_) | key_)
    if !isnothing(val)
      kw.args[2] =
        :(default_kwargs($(esc(ex[:name])), $(ex[:args]...); $(prev_kwargs...)).$key)
    end
    push!(prev_kwargs, key)
    return kw
  end

  # Promote to the type domain if wanted
  if astypes
    new_ex[:args] = map(ex[:args]) do arg
      @capture(arg, name_::T_)
      return :($(name)::Type{<:$T})
    end
  end

  new_ex[:name] = :(ITensorNetworks.default_kwargs)
  new_ex[:args] = convert(Vector{Any}, ex[:args])

  new_ex[:args] = pushfirst!(new_ex[:args], :(::typeof($(esc(ex[:name])))))

  # Escape anything on the right-hand side of a keyword definition.
  new_ex[:kwargs] = map(new_ex[:kwargs]) do kw
    @capture(kw, (key_ = val_) | key_)
    if !isnothing(val)
      kw.args[2] = esc(val)
    end
    return kw
  end

  new_ex[:body] = :(return (; $(prev_kwargs...)))

  # Escape the actual function name
  ex[:name] = :($(esc(ex[:name])))

  rv = quote
    $(combinedef(ex))
    $(combinedef(new_ex))
  end

  return rv
end
