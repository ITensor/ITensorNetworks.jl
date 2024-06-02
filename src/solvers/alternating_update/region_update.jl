#ToDo: generalize beyond 2-site
#ToDo: remove concept of orthogonality center for generality
function current_ortho(sweep_plan, which_region_update)
  regions = first.(sweep_plan)
  region = regions[which_region_update]
  current_verts = support(region)
  if !isa(region, AbstractEdge) && length(region) == 1
    return only(current_verts)
  end
  if which_region_update == length(regions)
    # look back by one should be sufficient, but may be brittle?
    overlapping_vertex = only(
      intersect(current_verts, support(regions[which_region_update - 1]))
    )
    return overlapping_vertex
  else
    # look forward
    other_regions = filter(
      x -> !(issetequal(x, current_verts)), support.(regions[(which_region_update + 1):end])
    )
    # find the first region that has overlapping support with current region 
    ind = findfirst(x -> !isempty(intersect(support(x), support(region))), other_regions)
    if isnothing(ind)
      # look backward
      other_regions = reverse(
        filter(
          x -> !(issetequal(x, current_verts)),
          support.(regions[1:(which_region_update - 1)]),
        ),
      )
      ind = findfirst(x -> !isempty(intersect(support(x), support(region))), other_regions)
    end
    @assert !isnothing(ind)
    future_verts = union(support(other_regions[ind]))
    # return ortho_ceter as the vertex in current region that does not overlap with following one
    overlapping_vertex = intersect(current_verts, future_verts)
    nonoverlapping_vertex = only(setdiff(current_verts, overlapping_vertex))
    return nonoverlapping_vertex
  end
end

function region_update(
  projected_operator,
  state::AbstractTTN;
  outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  region_printer,
  (region_observer!),
)
  (region, region_kwargs) = sweep_plan[which_region_update]
  (;
    extracter,
    extracter_kwargs,
    updater,
    updater_kwargs,
    inserter,
    inserter_kwargs,
    transform_operator,
    transform_operator_kwargs,
    internal_kwargs,
  ) = region_kwargs

  # ToDo: remove orthogonality center on vertex for generality
  # region carries same information 
  ortho_vertex = current_ortho(sweep_plan, which_region_update)
  if !isnothing(transform_operator)
    projected_operator = transform_operator(
      state, projected_operator; outputlevel, transform_operator_kwargs...
    )
  end
  state, projected_operator, phi = extracter(
    state, projected_operator, region, ortho_vertex; extracter_kwargs..., internal_kwargs
  )
  # create references, in case solver does (out-of-place) modify PH or state
  state! = Ref(state)
  projected_operator! = Ref(projected_operator)
  # args passed by reference are supposed to be modified out of place
  phi, info = updater(
    phi;
    state!,
    projected_operator!,
    outputlevel,
    which_sweep,
    sweep_plan,
    which_region_update,
    updater_kwargs...,
    internal_kwargs,
  )
  state = state![]
  projected_operator = projected_operator![]
  # ToDo: implement noise term as updater
  #drho = nothing
  #ortho = "left"    #i guess with respect to ordered vertices that's valid but may be cleaner to use next_region logic
  #if noise > 0.0 && isforward(direction)
  #  drho = noise * noiseterm(PH, phi, ortho) # TODO: actually implement this for trees...
  # so noiseterm is a solver
  #end
  state, spec = inserter(
    state, phi, region, ortho_vertex; inserter_kwargs..., internal_kwargs
  )
  all_kwargs = (;
    which_region_update,
    sweep_plan,
    total_sweep_steps=length(sweep_plan),
    end_of_sweep=(which_region_update == length(sweep_plan)),
    state,
    region,
    which_sweep,
    spec,
    outputlevel,
    info...,
    region_kwargs...,
    internal_kwargs...,
  )
  update_observer!(region_observer!; all_kwargs...)
  !(isnothing(region_printer)) && region_printer(; all_kwargs...)
  return state, projected_operator
end

function region_update(projected_operators, state; outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  region_printer,
  (region_observer!))

  (region, region_kwargs) = sweep_plan[which_region_update]
  (;
    extracter,
    extracter_kwargs,
    updater,
    updater_kwargs,
    inserter,
    inserter_kwargs,
    transform_operator,
    transform_operator_kwargs,
    internal_kwargs,
  ) = region_kwargs
  ψOψ_bpcs, ψIψ_bpc = first(projected_operators), last(projected_operators)
  
  #Fix extracter, update and inserter to work with sum of ψOψ_bpcs
  local_state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts = extracter(state, ψOψ_bpcs, ψIψ_bpc, region; extracter_kwargs...)

  local_state, _ = updater(local_state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts; updater_kwargs...)

  state, ψOψ_bpcs, ψIψ_bpc, spec, info  = inserter(state, ψOψ_bpcs, ψIψ_bpc, local_state, region; inserter_kwargs...)

  all_kwargs = (;
    which_region_update,
    sweep_plan,
    total_sweep_steps=length(sweep_plan),
    end_of_sweep=(which_region_update == length(sweep_plan)),
    state,
    region,
    which_sweep,
    spec,
    outputlevel,
    info...,
    region_kwargs...,
    internal_kwargs...,
  )
  update_observer!(region_observer!; all_kwargs...)
  !(isnothing(region_printer)) && region_printer(; all_kwargs...)

  return state, (ψOψ_bpcs, ψIψ_bpc)
end
