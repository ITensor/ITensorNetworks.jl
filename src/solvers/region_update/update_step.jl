#ToDo: generalize beyond 2-site
function current_ortho(sweep_plan, which_region_update)
  region = first(sweep_plan[which_region_update])
  current_verts=support(region)
  regions = first.(sweep_plan)
  if !isa(region,AbstractEdge) && length(region)==1
    return only(current_verts)
  end
  if which_region_update == length(regions)
    # look back by one should be sufficient, but maybe brittle?
    overlapping_vertex=only(intersect(current_verts,support(regions[which_region_update-1])))
    return overlapping_vertex
  else
    # look forward
    other_regions=filter(x -> !(issetequal(x,current_verts)), support.(regions[which_region_update+1:end]))
    # find the first region that has overlapping support with current region 
    ind=findfirst(x -> !isempty(intersect(support(x),support(region))),other_regions)
    future_verts=union(support(other_regions[ind]))
    # return ortho_ceter as the vertex in current region that does not overlap with following one
    overlapping_vertex=intersect(current_verts,future_verts)
    nonoverlapping_vertex = only(setdiff(current_verts,overlapping_vertex))
    return nonoverlapping_vertex
  end
end


function region_update(
  projected_operator,
  state;
  outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  region_printer,
  (region_observer!),
)
  (region, region_kwargs) = sweep_plan[which_region_update]
  ortho=current_ortho(sweep_plan,which_region_update)
  (; extract, update, insert, internal_kwargs) = region_kwargs
  extracter, extracter_kwargs = extract
  updater, updater_kwargs = update
  inserter, inserter_kwargs = insert
  state, projected_operator, phi = extract_local_tensor(
    state, projected_operator, region, ortho; extracter_kwargs..., internal_kwargs
  )
  state! = Ref(state) # create references, in case solver does (out-of-place) modify PH or state
  projected_operator! = Ref(projected_operator)
  phi, info = updater(
    phi;
    state!,
    projected_operator!,
    outputlevel,
    which_sweep,
    sweep_plan,
    which_region_update,
    updater_kwargs,
    internal_kwargs,
  )  # args passed by reference are supposed to be modified out of place
  state = state![] # dereference
  projected_operator = projected_operator![]
  if !(phi isa ITensor && info isa NamedTuple)
    println("Solver returned the following types: $(typeof(phi)), $(typeof(info))")
    error("In alternating_update, solver must return an ITensor and a NamedTuple")
  end
  #haskey(region_kwargs,:normalize) && ( region_kwargs.normalize && (phi /= norm(phi)) )
  # ToDo: implement noise term as updater
  #drho = nothing
  #ortho = "left"    #i guess with respect to ordered vertices that's valid but may be cleaner to use next_region logic
  #if noise > 0.0 && isforward(direction)
  #  drho = noise * noiseterm(PH, phi, ortho) # TODO: actually implement this for trees...
  # so noiseterm is a solver
  #end

  state, spec = insert_local_tensor(state, phi, region, ortho; inserter_kwargs..., internal_kwargs)

  all_kwargs = (;
    cutoff,
    maxdim,
    mindim,
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
  )
  !(isnothing(region_observer!)) && update!(region_observer!; all_kwargs...)

  !(isnothing(region_printer)) && region(; all_kwargs...)
  return state, projected_operator
end
