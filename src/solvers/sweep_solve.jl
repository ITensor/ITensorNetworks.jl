
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
)
  # Don't compute the region iteration automatically as we wish to insert a callback.
  for _ in IncrementOnly(sweep_iterator)
    for _ in region_iterator(sweep_iterator)
      region_callback(sweep_iterator)
    end
    sweep_callback(sweep_iterator)
  end
  return problem(sweep_iterator)
end

# I suspect that `sweep_callback` is the more commonly used callback, so allow this to
# be set using the `do` syntax.
function sweep_solve(
  sweep_callback, sweep_iterator; region_callback=default_region_callback
)
  return sweep_solve(sweep_iterator; sweep_callback, region_callback)
end

function sweep_solve(
  each_region_iterator::EachRegion; region_callback=default_region_callback
)
  return sweep_solve(region_callback, each_region_iterator)
end
function sweep_solve(region_callback, each_region_iterator::EachRegion)
  for _ in each_region_iterator
    # I don't think it is obvious what object this particular callback should take,
    # but for now be consistant and pass the parent sweep iterator.
    sweep_iterator = each_region_iterator.parent
    region_callback(sweep_iterator)
  end
  return problem(each_region_iterator)
end
