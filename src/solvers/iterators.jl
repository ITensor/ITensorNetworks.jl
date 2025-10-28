"""
  abstract type AbstractNetworkIterator

A stateful iterator with two states: `increment!` and `compute!`. Each iteration begins
with a call to `increment!` before executing `compute!`, however the initial call to
`iterate` skips the `increment!` call as it is assumed the iterator is initalized such that 
this call is implict. Termination of the iterator is controlled by the function `done`.
"""
abstract type AbstractNetworkIterator end

# We use greater than or equals here as we increment the state at the start of the iteration
islaststep(iterator::AbstractNetworkIterator) = state(iterator) >= length(iterator)

function Base.iterate(iterator::AbstractNetworkIterator, init = true)
    # The assumption is that first "increment!" is implicit, therefore we must skip the
    # the termination check for the first iteration, i.e. `AbstractNetworkIterator` is not
    # defined when length < 1,
    init || islaststep(iterator) && return nothing
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
mutable struct RegionIterator{Problem, RegionPlan} <: AbstractNetworkIterator
    problem::Problem
    region_plan::RegionPlan
    which_region::Int
    const which_sweep::Int
    function RegionIterator(problem::P, region_plan::R, sweep::Int) where {P, R}
        if length(region_plan) == 0
            throw(BoundsError("Cannot construct a region iterator with 0 elements."))
        end
        return new{P, R}(problem, region_plan, 1, sweep)
    end
end

function RegionIterator(problem; sweep, sweep_kwargs...)
    plan = region_plan(problem; sweep_kwargs...)
    return RegionIterator(problem, plan, sweep)
end

function new_region_iterator(iterator::RegionIterator; sweep_kwargs...)
    return RegionIterator(iterator.problem; sweep_kwargs...)
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

function region_kwargs(region_iter::RegionIterator)
    _, kwargs = current_region_plan(region_iter)
    return kwargs
end
function region_kwargs(f::Function, iter::RegionIterator)
    return get(region_kwargs(iter), Symbol(f, :_kwargs), (;))
end

function prev_region(region_iter::RegionIterator)
    state(region_iter) <= 1 && return nothing
    prev, _ = region_iter.region_plan[region_iter.which_region - 1]
    return prev
end

function next_region(region_iter::RegionIterator)
    islaststep(region_iter) && return nothing
    next, _ = region_iter.region_plan[region_iter.which_region + 1]
    return next
end

#
# Functions associated with RegionIterator
#
function increment!(region_iter::RegionIterator)
    region_iter.which_region += 1
    return region_iter
end

function compute!(iter::RegionIterator)
    _, local_state = extract!(iter; region_kwargs(extract!, iter)...)
    _, local_state = update!(iter, local_state; region_kwargs(update!, iter)...)
    insert!(iter, local_state; region_kwargs(insert!, iter)...)

    return iter
end

region_plan(problem; sweep_kwargs...) = euler_sweep(state(problem); sweep_kwargs...)

#
# SweepIterator
#

mutable struct SweepIterator{Problem, Iter} <: AbstractNetworkIterator
    region_iter::RegionIterator{Problem}
    sweep_kwargs::Iterators.Stateful{Iter}
    which_sweep::Int
    function SweepIterator(problem::Prob, sweep_kwargs::Iter) where {Prob, Iter}
        stateful_sweep_kwargs = Iterators.Stateful(sweep_kwargs)
        first_state = Iterators.peel(stateful_sweep_kwargs)

        if isnothing(first_state)
            throw(BoundsError("Cannot construct a sweep iterator with 0 elements."))
        end

        first_kwargs, _ = first_state
        region_iter = RegionIterator(problem; sweep = 1, first_kwargs...)

        return new{Prob, Iter}(region_iter, stateful_sweep_kwargs, 1)
    end
end

islaststep(sweep_iter::SweepIterator) = isnothing(peek(sweep_iter.sweep_kwargs))

region_iterator(sweep_iter::SweepIterator) = sweep_iter.region_iter
problem(sweep_iter::SweepIterator) = problem(region_iterator(sweep_iter))

state(sweep_iter::SweepIterator) = sweep_iter.which_sweep
Base.length(sweep_iter::SweepIterator) = length(sweep_iter.sweep_kwargs)
function increment!(sweep_iter::SweepIterator)
    sweep_iter.which_sweep += 1
    sweep_kwargs, _ = Iterators.peel(sweep_iter.sweep_kwargs)
    update_region_iterator!(sweep_iter; sweep_kwargs...)
    return sweep_iter
end

function update_region_iterator!(iterator::SweepIterator; kwargs...)
    sweep = state(iterator)
    iterator.region_iter = new_region_iterator(iterator.region_iter; sweep, kwargs...)
    return iterator
end

function compute!(sweep_iter::SweepIterator)
    for _ in sweep_iter.region_iter
        # TODO: Is it sensible to execute the default region callback function?
    end
    return
end

# More basic constructor where sweep_kwargs are constant throughout sweeps
function SweepIterator(problem, nsweeps::Int; sweep_kwargs...)
    # Initialize this to an empty RegionIterator
    sweep_kwargs_iter = Iterators.repeated(sweep_kwargs, nsweeps)
    return SweepIterator(problem, sweep_kwargs_iter)
end
