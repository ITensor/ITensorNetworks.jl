"""
  struct PauseAfterIncrement{S<:AbstractNetworkIterator}

Iterator wrapper whos `compute!` function simply returns itself, doing nothing in the 
process. This allows one to manually call a custom `compute!` or insert their own code it in
the loop body in place of `compute!`.
"""
struct NoComputeStep{S<:AbstractNetworkIterator} <: AbstractNetworkIterator
  parent::S
end

islaststep(adapter::NoComputeStep) = islaststep(adapter.parent)
state(adapter::NoComputeStep) = state(adapter.parent)
increment!(adapter::NoComputeStep) = increment!(adapter.parent)
compute!(adapter::NoComputeStep) = adapter

NoComputeStep(adapter::NoComputeStep) = adapter

"""
  struct EachRegion{SweepIterator} <: AbstractNetworkIterator

Adapter that flattens the each region iterator in the parent sweep iterator into a single
iterator, returning `region => kwargs`.
"""
struct EachRegion{SI<:SweepIterator} <: AbstractNetworkIterator
  parent::SI
end

# In keeping with Julia convention.
eachregion(iter::SweepIterator) = EachRegion(iter)

# Essential definitions
function islaststep(adapter::EachRegion)
  region_iter = region_iterator(adapter.parent)
  return islaststep(adapter.parent) && islaststep(region_iter)
end
function increment!(adapter::EachRegion)
  region_iter = region_iterator(adapter.parent)
  islaststep(region_iter) ? increment!(adapter.parent) : increment!(region_iter)
  return adapter
end
function compute!(adapter::EachRegion)
  region_iter = region_iterator(adapter.parent)
  compute!(region_iter)
  return current_region_plan(region_iter)
end

end
