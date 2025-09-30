"""
  abstract type AbstractNetworkIterator

A stateful iterator with two states: `increment!` and `compute!`. Each iteration begins
with a call to `increment!` before executing `compute!`, however the initial call to
`iterate` skips the `increment!` call as it is assumed the iterator is initalized such that 
this call is implict. Termination of the iterator is controlled by the function `done`.
"""
abstract type AbstractNetworkIterator end

# We use greater than or equals here as we increment the state at the start of the iteration
laststep(NI::AbstractNetworkIterator) = state(NI) >= length(NI)

function Base.iterate(NI::AbstractNetworkIterator, init=true)
  laststep(NI) && return nothing
  # We seperate increment! from step! and demand that any AbstractNetworkIterator *must*
  # define a method for increment! This way we avoid cases where one may wish to nest
  # calls to different step! methods accidentaly incrementing multiple times.
  init || increment!(NI)
  rv = compute!(NI)
  return rv, false
end

function increment! end
compute!(NI::AbstractNetworkIterator) = NI

step!(NI::AbstractNetworkIterator) = step!(identity, NI)
function step!(f, NI::AbstractNetworkIterator)
  compute!(NI)
  f(NI)
  increment!(NI)
  return NI
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

state(R::RegionIterator) = R.which_region
Base.length(R::RegionIterator) = length(R.region_plan)

problem(R::RegionIterator) = R.problem

current_region_plan(R::RegionIterator) = R.region_plan[R.which_region]

function current_region(R::RegionIterator)
  region, _ = current_region_plan(R)
  return region
end

function current_region_kwargs(R::RegionIterator)
  _, kwargs = current_region_plan(R)
  return kwargs
end

function previous_region(R::RegionIterator)
  state(R) <= 1 && return nothing
  prev, _ = R.region_plan[R.which_region - 1]
  return prev
end

function next_region(R::RegionIterator)
  is_last_region(R) && return nothing
  next, _ = R.region_plan[R.which_region + 1]
  return next
end
is_last_region(R::RegionIterator) = length(R) === state(R)

#
# Functions associated with RegionIterator
#

function compute!(R::RegionIterator)
  region_kwargs = current_region_kwargs(R)
  R.problem = region_step(R; region_kwargs...)
  return R
end
function increment!(R::RegionIterator)
  R.which_region += 1
  return R
end

function RegionIterator(problem; sweep, sweep_kwargs...)
  plan = region_plan(problem; sweep, sweep_kwargs...)
  return RegionIterator(problem, plan, sweep)
end

function region_step(
  region_iterator; extract_kwargs=(;), update_kwargs=(;), insert_kwargs=(;), kws...
)
  prob = problem(region_iterator)

  sweep = region_iterator.sweep

  prob, local_state = extract(prob, region_iterator; extract_kwargs..., sweep, kws...)
  prob, local_state = update(prob, local_state, region_iterator; update_kwargs..., kws...)
  prob = insert(prob, local_state, region_iterator; sweep, insert_kwargs..., kws...)
  return prob
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

laststep(SR::SweepIterator) = isnothing(peek(SR.sweep_kws))

region_iterator(S::SweepIterator) = S.region_iter
problem(S::SweepIterator) = problem(region_iterator(S))

state(SR::SweepIterator) = SR.which_sweep
Base.length(S::SweepIterator) = length(S.sweep_kws)
function increment!(SR::SweepIterator)
  SR.which_sweep += 1
  sweep_kwargs, _ = Iterators.peel(SR.sweep_kws)
  SR.region_iter = RegionIterator(problem(SR); sweep=state(SR), sweep_kwargs...)
  return SR
end

function compute!(SR::SweepIterator)
  for _ in SR.region_iter
    # TODO: Is it sensible to execute the default region callback function?
  end
end

# More basic constructor where sweep_kwargs are constant throughout sweeps
function SweepIterator(problem, nsweeps::Int; sweep_kwargs...)
  # Initialize this to an empty RegionIterator
  sweep_kwargs_iter = Iterators.repeated(sweep_kwargs, nsweeps)
  return SweepIterator(problem, sweep_kwargs_iter)
end
