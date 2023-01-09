function tdvp_step(
  order::TDVPOrder, solver, PH, time_step::Number, psi::MPS; current_time=0.0, kwargs...
)
  orderings = ITensorNetworks.orderings(order)
  sub_time_steps = ITensorNetworks.sub_time_steps(order)
  sub_time_steps *= time_step
  info = nothing
  for substep in 1:length(sub_time_steps)
    psi, PH, info = tdvp_sweep(
      orderings[substep], solver, PH, sub_time_steps[substep], psi; current_time, kwargs...
    )
    current_time += sub_time_steps[substep]
  end
  return psi, PH, info
end

isforward(direction::Base.ForwardOrdering) = true
isforward(direction::Base.ReverseOrdering) = false
isreverse(direction) = !isforward(direction)

function sweep_bonds(direction::Base.ForwardOrdering, n::Int; ncenter::Int)
  return 1:(n - ncenter + 1)
end

function sweep_bonds(direction::Base.ReverseOrdering, n::Int; ncenter::Int)
  return reverse(sweep_bonds(Base.Forward, n; ncenter))
end

is_forward_done(direction::Base.ForwardOrdering, b, n; ncenter) = (b + ncenter - 1 == n)
is_forward_done(direction::Base.ReverseOrdering, b, n; ncenter) = false
is_reverse_done(direction::Base.ForwardOrdering, b, n; ncenter) = false
is_reverse_done(direction::Base.ReverseOrdering, b, n; ncenter) = (b == 1)
function is_half_sweep_done(direction, b, n; ncenter)
  return is_forward_done(direction, b, n; ncenter) ||
         is_reverse_done(direction, b, n; ncenter)
end

function tdvp_sweep(
  direction::Base.Ordering, solver, PH, time_step::Number, psi::MPS; kwargs...
)
  PH = copy(PH)
  psi = copy(psi)
  if length(psi) == 1
    error(
      "`tdvp` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.exponentiate`, etc.",
    )
  end
  nsite::Int = get(kwargs, :nsite, 2)
  reverse_step::Bool = get(kwargs, :reverse_step, true)
  normalize::Bool = get(kwargs, :normalize, false)
  which_decomp::Union{String,Nothing} = get(kwargs, :which_decomp, nothing)
  svd_alg::String = get(kwargs, :svd_alg, "divide_and_conquer")
  observer = get(kwargs, :observer!, NoObserver())
  outputlevel = get(kwargs, :outputlevel, 0)
  sw = get(kwargs, :sweep, 1)
  current_time = get(kwargs, :current_time, 0.0)
  maxdim::Integer = get(kwargs, :maxdim, typemax(Int))
  mindim::Integer = get(kwargs, :mindim, 1)
  cutoff::Real = get(kwargs, :cutoff, 1E-16)
  noise::Real = get(kwargs, :noise, 0.0)

  N = length(psi)
  set_nsite!(PH, nsite)
  if isforward(direction)
    if !isortho(psi) || orthocenter(psi) != 1
      orthogonalize!(psi, 1)
    end
    @assert isortho(psi) && orthocenter(psi) == 1
    position!(PH, psi, 1)
  elseif isreverse(direction)
    if !isortho(psi) || orthocenter(psi) != N - nsite + 1
      orthogonalize!(psi, N - nsite + 1)
    end
    @assert(isortho(psi) && (orthocenter(psi) == N - nsite + 1))
    position!(PH, psi, N - nsite + 1)
  end
  maxtruncerr = 0.0
  info = nothing
  for b in sweep_bonds(direction, N; ncenter=nsite)
    current_time, maxtruncerr, spec, info = tdvp_site_update!(
      solver,
      PH,
      psi,
      b;
      nsite,
      reverse_step,
      current_time,
      outputlevel,
      time_step,
      normalize,
      direction,
      noise,
      which_decomp,
      svd_alg,
      cutoff,
      maxdim,
      mindim,
      maxtruncerr,
    )
    if outputlevel >= 2
      if nsite == 1
        @printf("Sweep %d, direction %s, bond (%d,) \n", sw, direction, b)
      elseif nsite == 2
        @printf("Sweep %d, direction %s, bond (%d,%d) \n", sw, direction, b, b + 1)
      end
      print("  Truncated using")
      @printf(" cutoff=%.1E", cutoff)
      @printf(" maxdim=%.1E", maxdim)
      print(" mindim=", mindim)
      print(" current_time=", round(current_time; digits=3))
      println()
      if spec != nothing
        @printf(
          "  Trunc. err=%.2E, bond dimension %d\n", spec.truncerr, dim(linkind(psi, b))
        )
      end
      flush(stdout)
    end
    update!(
      observer;
      psi,
      bond=b,
      sweep=sw,
      half_sweep=isforward(direction) ? 1 : 2,
      spec,
      outputlevel,
      half_sweep_is_done=is_half_sweep_done(direction, b, N; ncenter=nsite),
      current_time,
      info,
    )
  end
  # Just to be sure:
  normalize && normalize!(psi)
  return psi, PH, TDVPInfo(maxtruncerr)
end

function tdvp_site_update!(
  solver,
  PH,
  psi,
  b;
  nsite,
  reverse_step,
  current_time,
  outputlevel,
  time_step,
  normalize,
  direction,
  noise,
  which_decomp,
  svd_alg,
  cutoff,
  maxdim,
  mindim,
  maxtruncerr,
)
  return tdvp_site_update!(
    Val(nsite),
    Val(reverse_step),
    solver,
    PH,
    psi,
    b;
    current_time,
    outputlevel,
    time_step,
    normalize,
    direction,
    noise,
    which_decomp,
    svd_alg,
    cutoff,
    maxdim,
    mindim,
    maxtruncerr,
  )
end

function tdvp_site_update!(
  nsite_val::Val{1},
  reverse_step_val::Val{false},
  solver,
  PH,
  psi,
  b;
  current_time,
  outputlevel,
  time_step,
  normalize,
  direction,
  noise,
  which_decomp,
  svd_alg,
  cutoff,
  maxdim,
  mindim,
  maxtruncerr,
)
  N = length(psi)
  nsite = 1
  # Do 'forwards' evolution step
  set_nsite!(PH, nsite)
  position!(PH, psi, b)
  phi1 = psi[b]
  phi1, info = solver(PH, time_step, phi1; current_time, outputlevel)
  current_time += time_step
  normalize && (phi1 /= norm(phi1))
  spec = nothing
  psi[b] = phi1
  if !is_half_sweep_done(direction, b, N; ncenter=nsite)
    # Move ortho center
    Δ = (isforward(direction) ? +1 : -1)
    orthogonalize!(psi, b + Δ)
  end
  return current_time, maxtruncerr, spec, info
end

function tdvp_site_update!(
  nsite_val::Val{1},
  reverse_step_val::Val{true},
  solver,
  PH,
  psi,
  b;
  current_time,
  outputlevel,
  time_step,
  normalize,
  direction,
  noise,
  which_decomp,
  svd_alg,
  cutoff,
  maxdim,
  mindim,
  maxtruncerr,
)
  N = length(psi)
  nsite = 1
  # Do 'forwards' evolution step
  set_nsite!(PH, nsite)
  position!(PH, psi, b)
  phi1 = psi[b]
  phi1, info = solver(PH, time_step, phi1; current_time, outputlevel)
  current_time += time_step
  normalize && (phi1 /= norm(phi1))
  spec = nothing
  psi[b] = phi1
  if !is_half_sweep_done(direction, b, N; ncenter=nsite)
    # Do backwards evolution step
    b1 = (isforward(direction) ? b + 1 : b)
    Δ = (isforward(direction) ? +1 : -1)
    uinds = uniqueinds(phi1, psi[b + Δ])
    U, S, V = svd(phi1, uinds)
    psi[b] = U
    phi0 = S * V
    if isforward(direction)
      ITensors.setleftlim!(psi, b)
    elseif isreverse(direction)
      ITensors.setrightlim!(psi, b)
    end
    set_nsite!(PH, nsite - 1)
    position!(PH, psi, b1)
    phi0, info = solver(PH, -time_step, phi0; current_time, outputlevel)
    current_time -= time_step
    normalize && (phi0 ./= norm(phi0))
    psi[b + Δ] = phi0 * psi[b + Δ]
    if isforward(direction)
      ITensors.setrightlim!(psi, b + Δ + 1)
    elseif isreverse(direction)
      ITensors.setleftlim!(psi, b + Δ - 1)
    end
    set_nsite!(PH, nsite)
  end
  return current_time, maxtruncerr, spec, info
end

function tdvp_site_update!(
  nsite_val::Val{2},
  reverse_step_val::Val{false},
  solver,
  PH,
  psi,
  b;
  current_time,
  outputlevel,
  time_step,
  normalize,
  direction,
  noise,
  which_decomp,
  svd_alg,
  cutoff,
  maxdim,
  mindim,
  maxtruncerr,
)
  N = length(psi)
  nsite = 2
  # Do 'forwards' evolution step
  set_nsite!(PH, nsite)
  position!(PH, psi, b)
  phi1 = psi[b] * psi[b + 1]
  phi1, info = solver(PH, time_step, phi1; current_time, outputlevel)
  current_time += time_step
  normalize && (phi1 /= norm(phi1))
  spec = nothing
  ortho = isforward(direction) ? "left" : "right"
  drho = nothing
  if noise > 0.0 && isforward(direction)
    drho = noise * noiseterm(PH, phi, ortho)
  end
  spec = replacebond!(
    psi,
    b,
    phi1;
    maxdim,
    mindim,
    cutoff,
    eigen_perturbation=drho,
    ortho=ortho,
    normalize,
    which_decomp,
    svd_alg,
  )
  maxtruncerr = max(maxtruncerr, spec.truncerr)
  return current_time, maxtruncerr, spec, info
end

function tdvp_site_update!(
  nsite_val::Val{2},
  reverse_step_val::Val{true},
  solver,
  PH,
  psi,
  b;
  current_time,
  outputlevel,
  time_step,
  normalize,
  direction,
  noise,
  which_decomp,
  svd_alg,
  cutoff,
  maxdim,
  mindim,
  maxtruncerr,
)
  N = length(psi)
  nsite = 2
  # Do 'forwards' evolution step
  set_nsite!(PH, nsite)
  position!(PH, psi, b)
  phi1 = psi[b] * psi[b + 1]
  phi1, info = solver(PH, time_step, phi1; current_time, outputlevel)
  current_time += time_step
  normalize && (phi1 /= norm(phi1))
  spec = nothing
  ortho = isforward(direction) ? "left" : "right"
  drho = nothing
  if noise > 0.0 && isforward(direction)
    drho = noise * noiseterm(PH, phi, ortho)
  end
  spec = replacebond!(
    psi,
    b,
    phi1;
    maxdim,
    mindim,
    cutoff,
    eigen_perturbation=drho,
    ortho=ortho,
    normalize,
    which_decomp,
    svd_alg,
  )
  maxtruncerr = max(maxtruncerr, spec.truncerr)
  if !is_half_sweep_done(direction, b, N; ncenter=nsite)
    # Do backwards evolution step
    b1 = (isforward(direction) ? b + 1 : b)
    Δ = (isforward(direction) ? +1 : -1)
    phi0 = psi[b1]
    set_nsite!(PH, nsite - 1)
    position!(PH, psi, b1)
    phi0, info = solver(PH, -time_step, phi0; current_time, outputlevel)
    current_time -= time_step
    normalize && (phi0 ./= norm(phi0))
    psi[b1] = phi0
    set_nsite!(PH, nsite)
  end
  return current_time, maxtruncerr, spec, info
end

function tdvp_site_update!(
  ::Val{nsite},
  ::Val{reverse_step},
  solver,
  PH,
  psi,
  b;
  current_time,
  outputlevel,
  time_step,
  normalize,
  direction,
  noise,
  which_decomp,
  svd_alg,
  cutoff,
  maxdim,
  mindim,
  maxtruncerr,
) where {nsite,reverse_step}
  return error(
    "`tdvp` with `nsite=$nsite` and `reverse_step=$reverse_step` not implemented."
  )
end
