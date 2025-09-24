
function default_region_callback(sweep_iterator; kwargs...)
  return sweep_iterator
end
function default_sweep_callback(sweep_iterator; kwargs...)
  return sweep_iterator
end
# In this implementation the function `sweep_solve` is essentially just a wrapper around 
# the iterate interface that allows one to pass callbacks.
function sweep_solve(
  sweep_iterator;
  sweep_callback=default_sweep_callback,
  region_callback=default_region_callback,
  outputlevel=0,
)
  # Don't compute the region iteration automatically as we wish to insert a callback.
  for _ in PauseAfterIncrement(sweep_iterator)
    for _ in region_iterator(sweep_iterator)
      region_callback(sweep_iterator; outputlevel=outputlevel)
    end
    sweep_callback(sweep_iterator; outputlevel=outputlevel)
  end
  return problem(sweep_iterator)
end

# I suspect that `sweep_callback` is the more commonly used callback, so allow this to
# be set using the `do` syntax.
function sweep_solve(sweep_callback, sweep_iterator; kwargs...)
  return sweep_solve(sweep_iterator; sweep_callback=sweep_callback, kwargs...)
end
