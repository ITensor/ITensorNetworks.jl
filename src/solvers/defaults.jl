default_outputlevel() = 0
default_extractor() = extract_local_tensor
default_inserter() = insert_local_tensor

function default_region_printer(;
    cutoff,
    maxdim,
    mindim,
    outputlevel,
    state,
    sweep_plan,
    spec,
    which_region_update,
    which_sweep,
    kwargs...,
  )
    if outputlevel >= 2
      region = first(sweep_plan[which_region_update])
      @printf("Sweep %d, region=%s \n", which_sweep, region)
      print("  Truncated using")
      @printf(" cutoff=%.1E", cutoff)
      @printf(" maxdim=%d", maxdim)
      @printf(" mindim=%d", mindim)
      println()
      if spec != nothing
        @printf(
          "  Trunc. err=%.2E, bond dimension %d\n",
          spec.truncerr,
          linkdim(state, edgetype(state)(region...))
        )
      end
      flush(stdout)
    end
  end
  