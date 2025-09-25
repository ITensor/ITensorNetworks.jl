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

"""
  struct EachRegion{RegionIterator} <: AbstractNetworkIterator

Wapper adapter that returns a tuple (region, kwargs) at each step rather than the iterator
itself.
"""
struct EachRegion{R<:RegionIterator} <: AbstractNetworkIterator
  parent::R
end

# Essential definitions
Base.length(ER::EachRegion) = length(ER.parent)
state(ER::EachRegion) = state(ER.parent)
increment!(ER::EachRegion) = state(ER.parent)

function compute!(ER::EachRegion)
  # Do the usual compute! for RegionIterator
  compute!(ER.parent)
  # But now lets return something useful
  return current_region_plan(ER)
end
