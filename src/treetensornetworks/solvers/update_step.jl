
function default_sweep_regions(nsite, graph::AbstractGraph; kwargs...)
  return vcat(
    [
      half_sweep(
        direction(half),
        graph,
        make_region;
        nsite,
        region_args=(; half_sweep=half),
        kwargs...,
      ) for half in 1:2
    ]...,
  )
end

function step_printer(;
  cutoff, maxdim, mindim, outputlevel::Int=0, psi, region, spec, sweep_step
)
  if outputlevel >= 2
    @printf("Sweep %d, region=%s \n", sweep, region)
    print("  Truncated using")
    @printf(" cutoff=%.1E", cutoff)
    @printf(" maxdim=%d", maxdim)
    @printf(" mindim=%d", mindim)
    println()
    if spec != nothing
      @printf(
        "  Trunc. err=%.2E, bond dimension %d\n",
        spec.truncerr,
        linkdim(psi, edgetype(psi)(region...))
      )
    end
    flush(stdout)
  end
end

function update_step(
  solver,
  PH,
  psi::AbstractTTN;
  normalize::Bool=false,
  nsite::Int=2,
  step_printer=step_printer,
  (step_observer!)=observer(),
  sweep::Int=1,
  sweep_regions=default_sweep_regions(nsite, psi),
  kwargs...,
)
  PH = copy(PH)
  psi = copy(psi)

  insert_function!(step_observer!, "step_printer" => step_printer)

  # Append empty namedtuple to each element if not already present
  # (Needed to handle user-provided sweep_regions)
  sweep_regions = append_missing_namedtuple.(to_tuple.(sweep_regions))

  if nv(psi) == 1
    error(
      "`alternating_update` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.exponentiate`, etc.",
    )
  end

  for (sweep_step, (region, step_kwargs)) in enumerate(sweep_regions)
    psi, PH = local_update(
      solver,
      PH,
      psi,
      region;
      normalize,
      step_kwargs,
      step_observer!,
      sweep,
      sweep_regions,
      sweep_step,
      kwargs...,
    )
  end

  select!(step_observer!, Observers.DataFrames.Not("step_printer")) # remove step_printer
  # Just to be sure:
  normalize && normalize!(psi)

  return psi, PH
end

#
# Here extract_local_tensor and insert_local_tensor
# are essentially inverse operations, adapted for different kinds of 
# algorithms and networks.
#
# In the simplest case, exact_local_tensor contracts together a few
# tensors of the network and returns the result, while 
# insert_local_tensors takes that tensor and factorizes it back
# apart and puts it back into the network.
#

function extract_local_tensor(psi::AbstractTTN, pos::Vector)
  return psi, prod(psi[v] for v in pos)
end

function extract_local_tensor(psi::AbstractTTN, e::NamedEdge)
  left_inds = uniqueinds(psi, e)
  U, S, V = svd(psi[src(e)], left_inds; lefttags=tags(psi, e), righttags=tags(psi, e))
  psi[src(e)] = U
  return psi, S * V
end

# sort of multi-site replacebond!; TODO: use dense TTN constructor instead
function insert_local_tensor(
  psi::AbstractTTN,
  phi::ITensor,
  pos::Vector;
  normalize=false,
  # factorize kwargs
  maxdim=nothing,
  mindim=nothing,
  cutoff=nothing,
  which_decomp=nothing,
  eigen_perturbation=nothing,
  ortho=nothing,
)
  spec = nothing
  for (v, vnext) in IterTools.partition(pos, 2, 1)
    e = edgetype(psi)(v, vnext)
    indsTe = inds(psi[v])
    L, phi, spec = factorize(
      phi,
      indsTe;
      tags=tags(psi, e),
      maxdim,
      mindim,
      cutoff,
      which_decomp,
      eigen_perturbation,
      ortho,
    )
    psi[v] = L
    eigen_perturbation = nothing # TODO: fix this
  end
  psi[last(pos)] = phi
  psi = set_ortho_center(psi, [last(pos)])
  @assert isortho(psi) && only(ortho_center(psi)) == last(pos)
  normalize && (psi[last(pos)] ./= norm(psi[last(pos)]))
  # TODO: return maxtruncerr, will not be correct in cases where insertion executes multiple factorizations
  return psi, spec
end

function insert_local_tensor(psi::AbstractTTN, phi::ITensor, e::NamedEdge; kwargs...)
  psi[dst(e)] *= phi
  psi = set_ortho_center(psi, [dst(e)])
  return psi, nothing
end

#TODO: clean this up:
current_ortho(::Type{<:Vector{<:V}}, st) where {V} = first(st)
current_ortho(::Type{NamedEdge{V}}, st) where {V} = src(st)
current_ortho(st) = current_ortho(typeof(st), st)

function local_update(
  solver,
  PH,
  psi,
  region;
  normalize,
  noise,
  cutoff::AbstractFloat=1E-16,
  maxdim::Int=typemax(Int),
  mindim::Int=1,
  outputlevel::Int=0,
  step_kwargs=NamedTuple(),
  step_observer!,
  sweep,
  sweep_regions,
  sweep_step,
  solver_kwargs...,
)
  psi = orthogonalize(psi, current_ortho(region))
  psi, phi = extract_local_tensor(psi, region)

  nsites = (region isa AbstractEdge) ? 0 : length(region)
  PH = set_nsite(PH, nsites)
  PH = position(PH, psi, region)
  phi, info = solver(PH, phi; normalize, region, step_kwargs..., solver_kwargs...)
  if !(phi isa ITensor && info isa NamedTuple)
    println("Solver returned the following types: $(typeof(phi)), $(typeof(info))")
    error("In alternating_update, solver must return an ITensor and a NamedTuple")
  end
  normalize && (phi /= norm(phi))

  drho = nothing
  ortho = "left"
  #if noise > 0.0 && isforward(direction)
  #  drho = noise * noiseterm(PH, phi, ortho) # TODO: actually implement this for trees...
  #end

  psi, spec = insert_local_tensor(
    psi, phi, region; eigen_perturbation=drho, ortho, normalize, maxdim, mindim, cutoff
  )

  update!(
    step_observer!;
    cutoff,
    maxdim,
    mindim,
    sweep_step,
    total_sweep_steps=length(sweep_regions),
    end_of_sweep=(sweep_step == length(sweep_regions)),
    psi,
    region,
    sweep,
    spec,
    outputlevel,
    info...,
    step_kwargs...,
  )
  return psi, PH
end
