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
  error("Not implemented")
  return problem_cache
end
