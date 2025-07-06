
region_callback(problem; kws...) = nothing

sweep_callback(problem; kws...) = nothing

function sweep_printer(problem; outputlevel, sweep, nsweeps, kws...)
  if outputlevel >= 1
    println("Done with sweep $sweep/$nsweeps")
  end
end

function sweep_solve(
  sweep_iterator;
  outputlevel=0,
  region_callback=region_callback,
  sweep_callback=sweep_callback,
  sweep_printer=sweep_printer,
  kwargs...,
)
  for (sweep, region_iter) in enumerate(sweep_iterator)
    for (region, region_kwargs) in region_tuples(region_iter)
      region_callback(
        problem(region_iter);
        nsweeps=length(sweep_iterator),
        outputlevel,
        region,
        region_kwargs,
        sweep,
        kwargs...,
      )
    end
    sweep_callback(
      region_iter; nsweeps=length(sweep_iterator), outputlevel, sweep, kwargs...
    )
    sweep_printer(
      region_iter; nsweeps=length(sweep_iterator), outputlevel, sweep, kwargs...
    )
  end
  return problem(sweep_iterator)
end
