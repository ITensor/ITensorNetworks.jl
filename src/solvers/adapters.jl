
#
# RepeatIterator
#
# An "iterator of iterators", plugging in initialization
# arguments into an `init` function each repetition
#

mutable struct RepeatIterator{Iterator}
  iterator::Iterator
  keyword_args::Vector{<:NamedTuple}
end

function RepeatIterator(iter, n::Integer, kwargs::NamedTuple)
  return RepeatIterator(iter, fill(kwargs, n))
end

#
# Version 1: output the initialized iterator each time
#
function Base.iterate(R::RepeatIterator)
  inner_next = iterate(R.iterator)
  isnothing(inner_next) && return nothing
  (item, inner_state) = inner_next
  return item, (1, inner_state)
end
function Base.iterate(R::RepeatIterator, state)
  state = Base.iterate(keyword_args)
  isnothing(state) && return nothing
  (kwargs, which) = state
  R.iterator = init(R.iterator; kwargs...)
  return R, which
end

##
## Version 2: output the initialized iterator each time
##
#function Base.iterate(R::RepeatIterator, state=1)
#  state = Base.iterate(keyword_args)
#  isnothing(state) && return nothing
#  (kwargs, which) = state
#  R.iterator = init(R.iterator; kwargs...)
#  return R, which
#end

#
# TupleRegionIterator
#
# Adapts outputs to be (region, region_kwargs) tuples
#
# More generic design? maybe just assuming RegionIterator
# or its outputs implement some interface function that
# generates each tuple?
#

mutable struct TupleRegionIterator{RegionIter}
  region_iterator::RegionIter
end

region_iterator(T::TupleRegionIterator) = T.region_iterator

function Base.iterate(T::TupleRegionIterator, which=1)
  state = iterate(region_iterator(T), which)
  isnothing(state) && return nothing
  (current_region, region_kwargs) = current_region_plan(region_iterator(T))
  return (current_region, region_kwargs), last(state)
end

"""
  region_tuples(R::RegionIterator)

The `region_tuples` adapter converts a RegionIterator into an 
iterator which outputs a tuple of the form (current_region, current_region_kwargs)
at each step.
"""
region_tuples(R::RegionIterator) = TupleRegionIterator(R)
