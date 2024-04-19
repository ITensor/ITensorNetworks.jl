using SerializedElementArrays: disk
# Utilities for making it easier
# to define solvers (like ODE solvers)
# for TDVP
"""
    to_vec(x)

Transform `x` into a `Vector`. Returns the vector and a closure which inverts the
transformation.

Modeled after `FiniteDifferences.to_vec`:

https://github.com/JuliaDiff/FiniteDifferences.jl/blob/main/src/to_vec.jl
"""
to_vec(x) = error("Not implemented")

function to_vec(x::ITensor)
  function ITensor_from_vec(x_vec)
    return itensor(x_vec, inds(x))
  end
  return vec(array(x)), ITensor_from_vec
end

# Represents a time-dependent sum of terms:
#
# H(t) = f[1](t) * H0[1] + f[2](t) * H0[2] + …
#
struct TimeDependentSum{S,T}
  f::Vector{S}
  H0::T
end
TimeDependentSum(f::Vector, H0::ProjTTNSum) = TimeDependentSum(f, terms(H0))
Base.length(H::TimeDependentSum) = length(H.f)

function Base.:*(c::Number, H::TimeDependentSum)
  return TimeDependentSum([t -> c * fₙ(t) for fₙ in H.f], H.H0)
end
Base.:*(H::TimeDependentSum, c::Number) = c * H

# Calling a `TimeDependentOpSum` at a certain time like:
#
# H(t)
#
# Returns a `ScaledSum` at that time.
(H::TimeDependentSum)(t::Number) = ScaledSum([fₙ(t) for fₙ in H.f], H.H0)

# Represents the sum of scaled terms:
#
# H = coefficient[1] * H[1] + coefficient * H[2] + …
#
struct ScaledSum{S,T}
  coefficients::Vector{S}
  H::T
end
Base.length(H::ScaledSum) = length(H.coefficients)

# Apply the scaled sum of terms:
#
# H(ψ₀) = coefficient[1] * H[1](ψ₀) + coefficient[2] * H[2](ψ₀) + …
#
# onto ψ₀.
function (H::ScaledSum)(ψ₀)
  ψ = ITensor(inds(ψ₀))
  for n in 1:length(H)
    ψ += H.coefficients[n] * apply(H.H[n], ψ₀)
  end
  return permute(ψ, inds(ψ₀))
end

function cache_operator_to_disk(
  state,
  operator;
  # univeral kwarg signature
  outputlevel,
  # non-universal kwarg
  write_when_maxdim_exceeds,
)
  isnothing(write_when_maxdim_exceeds) && return operator
  m = maximum(edge_data(linkdims(state)))
  if m > write_when_maxdim_exceeds
    if outputlevel >= 2
      println(
        "write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxlinkdim = $(m), writing environment tensors to disk",
      )
    end
    operator = disk(operator)
  end
  return operator
end

#ToDo: Move? This belongs more into local_solvers
function compose_updaters(;kwargs...)
  funcs=values(kwargs)
  kwarg_symbols=Symbol.(keys(kwargs),"_kwargs")
  info_symbols=Symbol.(keys(kwargs),"_info")
  function composed_updater(init; kwargs...)
    info=(;)
    kwargs_for_updaters=map(x -> kwargs[x], kwarg_symbols)
    other_kwargs=Base.structdiff(kwargs,NamedTuple(kwarg_symbols .=> kwargs_for_updaters))
  
    for (func,kwargs_for_updater,info_symbol) in zip(funcs,kwargs_for_updaters,info_symbols)
      init, new_info=func(init;
      other_kwargs...,
      kwargs_for_updater...
      )
      #figure out a better way to handle composing the info?
      info=(;info...,NamedTuple((info_symbol=>new_info,))...)
    end
    return init, info
  end
  return composed_updater
end