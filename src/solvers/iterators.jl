
#
# sweep_iterator
#

function sweep_iterator(problem, sweep_kwargs_array)
  return [region_iterator(problem; sweep_kwargs...) for sweep_kwargs in sweep_kwargs_array]
end

function sweep_iterator(problem, nsweeps::Integer)
  return sweep_iterator(problem, Iterators.repeated((;), nsweeps))
end

#
# step_iterator
#

step_iterator(args...; kws...) = Iterators.flatten(sweep_iterator(args...; kws...))

#
# RegionIterator
#

@kwdef mutable struct RegionIterator{Problem,RegionPlan}
  problem::Problem
  region_plan::RegionPlan
  which_region_plan::Int = 1
  prev_region = nothing
  #extra_kwargs::NamedTuple = (;)
end

problem(R::RegionIterator) = R.problem
current_region_plan(R::RegionIterator) = R.region_plan[R.which_region_plan]
current_region(R::RegionIterator) = R.region_plan[R.which_region_plan][1]
region_kwargs(R::RegionIterator) = R.region_plan[R.which_region_plan][2]

function Base.iterate(R::RegionIterator, which=1)
  R.which_region_plan = which
  region_plan_state = iterate(R.region_plan, which)
  isnothing(region_plan_state) && return nothing
  (current_region, region_kwargs), next = region_plan_state

  region_iterator_action!(problem(R); region=current_region, prev_region=R.prev_region, region_kwargs...)
  R.prev_region = current_region
  return R, next
end

#
# Functions associated with RegionIterator
#

function region_iterator(problem; nsites=1, sweep_kwargs...)
  return RegionIterator(;
    problem, region_plan=region_plan(problem; nsites, sweep_kwargs...)
  )
end

function region_iterator_action!(
  problem; region, prev_region=nothing, extracter_kwargs=(;), updater_kwargs=(;), inserter_kwargs=(;), kwargs...
)
  local_tensor = extracter!(problem, region; extracter_kwargs..., kwargs...)
  local_tensor = prepare_subspace!(problem, local_tensor, region; prev_region, extracter_kwargs..., kwargs...)
  local_tensor = updater!(problem, local_tensor, region; updater_kwargs..., kwargs...)
  inserter!(problem, local_tensor, region; inserter_kwargs..., kwargs...)
  return
end

function region_plan(problem; nsites, sweep_kwargs...)
  return basic_region_plan(state(problem); nsites, sweep_kwargs...)
end
