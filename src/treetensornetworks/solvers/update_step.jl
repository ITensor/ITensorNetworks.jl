
function default_sweep_regions(nsites, graph::AbstractGraph; kwargs...)  ###move this to a different file, algorithmic level idea
  return vcat(
    [
      half_sweep(
        direction(half),
        graph,
        make_region;
        nsites,
        region_args=(; half_sweep=half),
        kwargs...,
      ) for half in 1:2
    ]...,
  )
end

function region_update_printer(;
  cutoff, maxdim, mindim, outputlevel::Int=0, psi, region_updates, spec, which_region_update, which_sweep,kwargs...
)
  if outputlevel >= 2
    region=first(region_updates[which_region_update])
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
        linkdim(psi, edgetype(psi)(region...))
      )
    end
    flush(stdout)
  end
end

function sweep_update(
  solver,
  PH,
  psi::AbstractTTN;
  normalize::Bool=false,      # ToDo: think about where to put the default, probably this default is best defined at algorithmic level
  outputlevel,
  region_update_printer=region_update_printer,
  (region_observer!)=observer(),  # ToDo: change name to region_observer! ?
  which_sweep::Int,
  sweep_params::NamedTuple,
  region_updates,# =default_sweep_regions(nsite, psi; reverse_step),   #move default up to algorithmic level
  updater_kwargs,
)
  insert_function!(region_observer!, "region_update_printer" => region_update_printer) #ToDo fix this

  # Append empty namedtuple to each element if not already present
  # (Needed to handle user-provided region_updates)
  region_updates = append_missing_namedtuple.(to_tuple.(region_updates))

  if nv(psi) == 1
    error(
      "`alternating_update` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.exponentiate`, etc.",
    )
  end
  
  for which_region_update in eachindex(region_updates)
    # merge sweep params in step_kwargs
    (region, region_kwargs)=region_updates[which_region_update]
    region_kwargs=merge(region_kwargs, sweep_params)    #in this case sweep params has precedence over step_kwargs --- correct behaviour?
    psi, PH = region_update(
      solver,
      PH,
      psi;
      normalize,
      outputlevel,
      which_sweep,
      region_updates,
      which_region_update,
      region_kwargs,
      region_observer!,
      updater_kwargs,
    )
  end

   select!(region_observer!, Observers.DataFrames.Not("region_update_printer")) # remove update_printer
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
# also can we entirely rely on directionality of edges by construction?
current_ortho(::Type{<:Vector{<:V}}, st) where {V} = first(st)
current_ortho(::Type{NamedEdge{V}}, st) where {V} = src(st)
current_ortho(st) = current_ortho(typeof(st), st)

function region_update(
  updater,
  PH,
  psi;
  normalize,
  outputlevel,
  which_sweep,
  region_updates,
  which_region_update,
  region_kwargs,
  region_observer!,
  #insertion_kwargs,  #ToDo: later
  #extraction_kwargs, #ToDo: implement later with possibility to pass custom extraction/insertion func (or code into func)
  updater_kwargs
)
  region=first(region_updates[which_region_update])
  psi = orthogonalize(psi, current_ortho(region))
  psi, phi = extract_local_tensor(psi, region;)
  nsites = (region isa AbstractEdge) ? 0 : length(region) #ToDo move into separate funtion
  PH = set_nsite(PH, nsites)
  PH = position(PH, psi, region)
  psi_ref! = Ref(psi) # create references, in case solver does (out-of-place) modify PH or psi
  PH_ref! = Ref(PH) 
  phi, info = updater(
    phi;
    psi_ref!,
    PH_ref!,
    outputlevel,
    which_sweep,
    region_updates,
    which_region_update,
    region_kwargs,
    updater_kwargs
  )  # args passed by reference are supposed to be modified out of place
  psi = psi_ref![] # dereference
  PH = PH_ref![]
  if !(phi isa ITensor && info isa NamedTuple)
    println("Solver returned the following types: $(typeof(phi)), $(typeof(info))")
    error("In alternating_update, solver must return an ITensor and a NamedTuple")
  end
  normalize && (phi /= norm(phi))

  drho = nothing
  ortho = "left"    #i guess with respect to ordered vertices that's valid but may be cleaner to use next_region logic
  #if noise > 0.0 && isforward(direction)
  #  drho = noise * noiseterm(PH, phi, ortho) # TODO: actually implement this for trees...
  # so noiseterm is a solver
  #end

  psi, spec = insert_local_tensor(
    psi, phi, region; eigen_perturbation=drho, ortho, normalize,
    maxdim=region_kwargs.maxdim,
    mindim=region_kwargs.mindim,
    cutoff=region_kwargs.cutoff
  )

  update!(
    region_observer!;
    cutoff,
    maxdim,
    mindim,
    which_region_update,
    region_updates,
    total_sweep_steps=length(region_updates),
    end_of_sweep=(which_region_update == length(region_updates)),
    psi,
    region,
    which_sweep,
    spec,
    outputlevel,
    info...,
    region_kwargs...,
  )
  return psi, PH
end
