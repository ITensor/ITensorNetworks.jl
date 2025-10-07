"""
  abstract type AbstractNetworkIterator

A stateful iterator with two states: `increment!` and `compute!`. Each iteration begins
with a call to `increment!` before executing `compute!`, however the initial call to
`iterate` skips the `increment!` call as it is assumed the iterator is initalized such that 
this call is implict. Termination of the iterator is controlled by the function `done`.
"""
abstract type AbstractNetworkIterator end

# We use greater than or equals here as we increment the state at the start of the iteration
laststep(iterator::AbstractNetworkIterator) = state(iterator) >= length(iterator)

function Base.iterate(iterator::AbstractNetworkIterator, init=true)
  laststep(iterator) && return nothing
  # We seperate increment! from step! and demand that any AbstractNetworkIterator *must*
  # define a method for increment! This way we avoid cases where one may wish to nest
  # calls to different step! methods accidentaly incrementing multiple times.
  init || increment!(iterator)
  rv = compute!(iterator)
  return rv, false
end

function increment! end
compute!(iterator::AbstractNetworkIterator) = iterator

step!(iterator::AbstractNetworkIterator) = step!(identity, iterator)
function step!(f, iterator::AbstractNetworkIterator)
  compute!(iterator)
  f(iterator)
  increment!(iterator)
  return iterator
end

#
# RegionIterator
#
"""
  struct RegionIterator{Problem, RegionPlan} <: AbstractNetworkIterator
"""
mutable struct RegionIterator{Problem,RegionPlan} <: AbstractNetworkIterator
  problem::Problem
  region_plan::RegionPlan
  const sweep::Int
  which_region::Int
  function RegionIterator(problem::P, region_plan::R, sweep::Int) where {P,R}
    return new{P,R}(problem, region_plan, sweep, 1)
  end
end

state(region_iter::RegionIterator) = region_iter.which_region
Base.length(region_iter::RegionIterator) = length(region_iter.region_plan)

problem(region_iter::RegionIterator) = region_iter.problem

function current_region_plan(region_iter::RegionIterator)
  return region_iter.region_plan[region_iter.which_region]
end

function current_region(region_iter::RegionIterator)
  region, _ = current_region_plan(region_iter)
  return region
end

function current_region_kwargs(region_iter::RegionIterator)
  _, kwargs = current_region_plan(region_iter)
  return kwargs
end

function prev_region(region_iter::RegionIterator)
  state(region_iter) <= 1 && return nothing
  prev, _ = region_iter.region_plan[region_iter.which_region - 1]
  return prev
end

function next_region(region_iter::RegionIterator)
  is_last_region(region_iter) && return nothing
  next, _ = region_iter.region_plan[region_iter.which_region + 1]
  return next
end
is_last_region(region_iter::RegionIterator) = length(region_iter) === state(region_iter)

#
# Functions associated with RegionIterator
#
function increment!(region_iter::RegionIterator)
  region_iter.which_region += 1
  return region_iter
end

function compute!(iter::RegionIterator)
  local_state = extract!(iter; current_kwargs(extract!, iter)...)
  local_state = update!(local_state, iter; current_kwargs(update!, iter)...)
  insert!(local_state, iter; current_kwargs(insert!, iter)...)

  return iter
end

function RegionIterator(problem; sweep, sweep_kwargs...)
  plan = region_plan(problem; sweep, sweep_kwargs...)
  return RegionIterator(problem, plan, sweep)
end

function region_plan(problem; kws...)
  return euler_sweep(state(problem); kws...)
end

#
# SweepIterator
#

mutable struct SweepIterator{Problem} <: AbstractNetworkIterator
  sweep_kws
  region_iter::RegionIterator{Problem}
  which_sweep::Int
  function SweepIterator(problem, sweep_kws)
    sweep_kws = Iterators.Stateful(sweep_kws)
    first_kwargs, _ = Iterators.peel(sweep_kws)
    region_iter = RegionIterator(problem; sweep=1, first_kwargs...)
    return new{typeof(problem)}(sweep_kws, region_iter, 1)
  end
end

laststep(sweep_iter::SweepIterator) = isnothing(peek(sweep_iter.sweep_kws))

region_iterator(sweep_iter::SweepIterator) = sweep_iter.region_iter
problem(sweep_iter::SweepIterator) = problem(region_iterator(sweep_iter))

state(sweep_iter::SweepIterator) = sweep_iter.which_sweep
Base.length(sweep_iter::SweepIterator) = length(sweep_iter.sweep_kws)
function increment!(sweep_iter::SweepIterator)
  sweep_iter.which_sweep += 1
  sweep_kwargs, _ = Iterators.peel(sweep_iter.sweep_kws)
  sweep_iter.region_iter = RegionIterator(
    problem(sweep_iter); sweep=state(sweep_iter), sweep_kwargs...
  )
  return sweep_iter
end

function compute!(sweep_iter::SweepIterator)
  for _ in sweep_iter.region_iter
    # TODO: Is it sensible to execute the default region callback function?
  end
end

# More basic constructor where sweep_kwargs are constant throughout sweeps
function SweepIterator(problem, nsweeps::Int; sweep_kwargs...)
  # Initialize this to an empty RegionIterator
  sweep_kwargs_iter = Iterators.repeated(sweep_kwargs, nsweeps)
  return SweepIterator(problem, sweep_kwargs_iter)
end
