function alternating_update(problem_cache, sweeps)
  for sweep in sweeps
    problem_cache = alternating_update_sweep(
      problem_cache,
      sweep,
    )
  end
  return problem_cache
end

function alternating_update_sweep(problem_cache, region_updates)
  for region_update in region_updates
    problem_cache = update_region(
      problem_cache,
      region_update,
    )
  end
  return problem_cache
end

function update_region(problem_cache, updates)
  @show updates
  @show updates.region
  region = updates.region
  # TODO: Call this `reduced_state` or `region_state`? Contract
  # together into a single ITensor or keep separate?
  for vertex in region
    # Get the tensors of the state within the region being updated.
    @show state(problem_cache, vertex)
  end
  solvers = updates.solvers
  for solver in solvers
    new_region_state = solver(region_state)
    problem_cache = insert_region_state(problem_cache, new_region_state)
  end
  error("Not implemented")
  return problem_cache
end
