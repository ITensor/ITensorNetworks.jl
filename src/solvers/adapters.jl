
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

"""
  struct PauseAfterIncrement{S<:AbstractNetworkIterator}

Iterator wrapper whos `compute!` function simply returns itself, doing nothing in the 
process. This allows one to manually call a custom `compute!` or insert their own code it in
the loop body in place of `compute!`.
"""
struct PauseAfterIncrement{S<:AbstractNetworkIterator} <: AbstractNetworkIterator
  parent::S
end

done(NC::PauseAfterIncrement) = done(NC.parent)
state(NC::PauseAfterIncrement) = state(NC.parent)
increment!(NC::PauseAfterIncrement) = increment!(NC.parent)
compute!(NC::PauseAfterIncrement) = NC

PauseAfterIncrement(NC::PauseAfterIncrement) = NC
