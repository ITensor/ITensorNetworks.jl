"""
  struct PauseAfterIncrement{S<:AbstractNetworkIterator}

Iterator wrapper whos `compute!` function simply returns itself, doing nothing in the 
process. This allows one to manually call a custom `compute!` or insert their own code it in
the loop body in place of `compute!`.
"""
struct NoComputeStep{S<:AbstractNetworkIterator} <: AbstractNetworkIterator
  parent::S
end

laststep(adapter::NoComputeStep) = laststep(adapter.parent)
state(adapter::NoComputeStep) = state(adapter.parent)
increment!(adapter::NoComputeStep) = increment!(adapter.parent)
compute!(adapter::NoComputeStep) = adapter

NoComputeStep(adapter::NoComputeStep) = adapter

"""
  struct EachRegion{RegionIterator} <: AbstractNetworkIterator

Wapper adapter that returns a tuple (region, kwargs) at each step rather than the iterator
itself.
"""
struct EachRegion{R<:RegionIterator} <: AbstractNetworkIterator
  parent::R
end

# Essential definitions
Base.length(adapter::EachRegion) = length(adapter.parent)
state(adapter::EachRegion) = state(adapter.parent)
increment!(adapter::EachRegion) = state(adapter.parent)

function compute!(adapter::EachRegion)
  # Do the usual compute! for RegionIterator
  compute!(adapter.parent)
  # But now lets return something useful
  return current_region_plan(adapter)
end
