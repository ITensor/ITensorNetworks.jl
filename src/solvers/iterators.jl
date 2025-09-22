#
# RegionIterator
#

@kwdef mutable struct RegionIterator{Problem,RegionPlan}
  problem::Problem
  region_plan::RegionPlan
  which_region::Int = 1
end

problem(R::RegionIterator) = R.problem
current_region_plan(R::RegionIterator) = R.region_plan[R.which_region]
current_region(R::RegionIterator) = current_region_plan(R)[1]
region_kwargs(R::RegionIterator) = current_region_plan(R)[2]
function previous_region(R::RegionIterator)
  return R.which_region == 1 ? nothing : R.region_plan[R.which_region - 1][1]
end
function next_region(R::RegionIterator)
  return if R.which_region == length(R.region_plan)
    nothing
  else
    R.region_plan[R.which_region + 1][1]
  end
end
is_last_region(R::RegionIterator) = isnothing(next_region(R))

function Base.iterate(R::RegionIterator, which=1)
  R.which_region = which
  region_plan_state = iterate(R.region_plan, which)
  isnothing(region_plan_state) && return nothing
  (current_region, region_kwargs), next = region_plan_state
  R.problem = region_step(problem(R), R; region_kwargs...)
  return R, next
end

#
# Functions associated with RegionIterator
#

function region_iterator(problem; sweep_kwargs...)
  return RegionIterator(; problem, region_plan=region_plan(problem; sweep_kwargs...))
end

function region_step(
  problem,
  region_iterator;
  extract_kwargs=(;),
  update_kwargs=(;),
  insert_kwargs=(;),
  sweep,
  kws...,
)
  problem, local_state = extract(problem, region_iterator; extract_kwargs..., sweep, kws...)
  problem, local_state = update(
    problem, local_state, region_iterator; update_kwargs..., kws...
  )
  problem = insert(problem, local_state, region_iterator; sweep, insert_kwargs..., kws...)
  return problem
end

function region_plan(problem; kws...)
  return euler_sweep(state(problem); kws...)
end

#
# SweepIterator
#

mutable struct SweepIterator{Problem}
  sweep_kws
  region_iter::RegionIterator{Problem}
  which_sweep::Int
end

problem(S::SweepIterator) = problem(S.region_iter)

Base.length(S::SweepIterator) = length(S.sweep_kws)

function Base.iterate(S::SweepIterator, which=nothing)
  if isnothing(which)
    sweep_kws_state = iterate(S.sweep_kws)
  else
    sweep_kws_state = iterate(S.sweep_kws, which)
  end
  isnothing(sweep_kws_state) && return nothing
  current_sweep_kws, next = sweep_kws_state

  if !isnothing(which)
    S.region_iter = region_iterator(
      problem(S.region_iter); sweep=S.which_sweep, current_sweep_kws...
    )
  end
  S.which_sweep += 1
  return S.region_iter, next
end

function sweep_iterator(problem, sweep_kws)
  region_iter = region_iterator(problem; sweep=1, first(sweep_kws)...)
  return SweepIterator(sweep_kws, region_iter, 1)
end

function sweep_iterator(problem, nsweeps::Integer; sweep_kws...)
  return sweep_iterator(problem, Iterators.repeated(sweep_kws, nsweeps))
end
