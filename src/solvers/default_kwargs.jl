using MacroTools: @capture, splitdef, combinedef, isdef

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
    @define_default_kwargs

Automatically define a `default_kwargs` method for a given function. This macro should
be applied before a function definition:
```
@define_default_kwargs function f(arg1::T1, arg2::T2, ...; kwargs...) 
  ...
end
```
The defined `default_kwargs` method takes the form
```
default_kwargs(::typeof(f), arg1::T1, arg2::T2, ...; kwargs...)
```
i.e. the function signature mirrors that of the function signature of `f`.
"""
macro define_default_kwargs(function_def)
  return default_kwargs_macro(function_def)
end

function default_kwargs_macro(function_def)
  if !isdef(function_def)
    throw(
      ArgumentError(
        "The @define_default_kwargs macro must be followed by a function definition"
      ),
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

  new_ex[:args] = convert(Vector{Any}, ex[:args])

  new_ex[:name] = :(ITensorNetworks.default_kwargs)
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

macro with_defaults(call_expr)
  if @capture(call_expr, (func_(args__; kwargs__)) | (func_(args__)))
    if isnothing(kwargs)
      kwargs = []
    end
    rv = quote
      $(esc(func))(
        $(esc.(args)...);
        default_kwargs($(esc(func)), $(esc.(args)...); $(esc.(kwargs)...))...,
      )
    end
    return rv
  else
    throw(ArgumentError("unable to parse function call expression, try including brackets in the macro call."))
  end
end
