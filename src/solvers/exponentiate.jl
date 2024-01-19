function exponentiate_updater(
  init;
  psi_ref!,
  PH_ref!,
  outputlevel,
  which_sweep,
  region_updates,
  which_region_update,
  region_kwargs,
  updater_kwargs,
)
  default_updater_kwargs = (;
    krylovdim=30,  #from here only solver kwargs
    maxiter=100,
    outputlevel=0,
    tol=1E-12,
    ishermitian=true,
    issymmetric=true,
    eager=true,
  )
  updater_kwargs = merge(default_updater_kwargs, updater_kwargs)  #last collection has precedence
  #H=copy(PH_ref![])
  H = PH_ref![] ###since we are not changing H we don't need the copy
  # let's test whether given region and sweep regions we can find out what the previous and next region were
  # this will be needed in subspace expansion
  #@show step_kwargs
  substep = get(region_kwargs, :substep, nothing)
  time_step = get(region_kwargs, :time_step, nothing)
  @assert !isnothing(time_step) && !isnothing(substep)
  region_ind = which_region_update
  next_region =
    region_ind == length(region_updates) ? nothing : first(region_updates[region_ind + 1])
  previous_region = region_ind == 1 ? nothing : first(region_updates[region_ind - 1])

  phi, exp_info = exponentiate(
    H,
    time_step,
    init;
    ishermitian=updater_kwargs.ishermitian,
    issymmetric=updater_kwargs.issymmetric,
    tol=updater_kwargs.tol,
    krylovdim=updater_kwargs.krylovdim,
    maxiter=updater_kwargs.maxiter,
    verbosity=updater_kwargs.outputlevel,
    eager=updater_kwargs.eager,
  )
  return phi, (; info=exp_info)
end
