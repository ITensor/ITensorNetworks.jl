#ToDo: Move elsewhere
default_extractor() = extract_local_tensor
default_inserter() = insert_local_tensor

function default_region_update_printer(;
  cutoff,
  maxdim,
  mindim,
  outputlevel::Int=0,
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

function sweep_update(
  projected_operator, state::AbstractTTN; outputlevel, which_sweep::Int, sweep_plan
)

  # Append empty namedtuple to each element if not already present
  # (Needed to handle user-provided region_updates)
  # todo: Hopefully not needed anymore
  sweep_plan = append_missing_namedtuple.(to_tuple.(sweep_plan))

  if nv(state) == 1
    error(
      "`alternating_update` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.exponentiate`, etc.",
    )
  end

  for which_region_update in eachindex(sweep_plan)
    state, projected_operator = region_update(
      projected_operator,
      state;
      which_sweep,
      sweep_plan,
      which_region_update,
      outputlevel, # ToDo      
    )
  end

  return state, projected_operator
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

function extract_prolog(state::AbstractTTN, region)
  return state = orthogonalize(state, current_ortho(region))
end

function extract_epilog(state::AbstractTTN, projected_operator, region)
  #nsites = (region isa AbstractEdge) ? 0 : length(region)
  #projected_operator = set_nsite(projected_operator, nsites) #not necessary
  projected_operator = position(projected_operator, state, region)
  return projected_operator   #should it return only projected_operator
end

function extract_local_tensor(
  state::AbstractTTN, projected_operator, pos::Vector; extract_kwargs...
)
  state = extract_prolog(state, pos)
  projected_operator = extract_epilog(state, projected_operator, pos)
  return state, projected_operator, prod(state[v] for v in pos)
end

function extract_local_tensor(
  state::AbstractTTN, projected_operator, e::AbstractEdge; extract_kwargs...
)
  state = extract_prolog(state, e)
  left_inds = uniqueinds(state, e)
  U, S, V = svd(state[src(e)], left_inds; lefttags=tags(state, e), righttags=tags(state, e))
  state[src(e)] = U
  projected_operator = extract_epilog(state, projected_operator, e)
  return state, projected_operator, S * V
end

# sort of multi-site replacebond!; TODO: use dense TTN constructor instead
function insert_local_tensor(
  state::AbstractTTN,
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
  kwargs...,
)
  spec = nothing
  for (v, vnext) in IterTools.partition(pos, 2, 1)
    e = edgetype(state)(v, vnext)
    indsTe = inds(state[v])
    L, phi, spec = factorize(
      phi,
      indsTe;
      tags=tags(state, e),
      maxdim,
      mindim,
      cutoff,
      which_decomp,
      eigen_perturbation,
      ortho,
    )
    state[v] = L
    eigen_perturbation = nothing # TODO: fix this
  end
  state[last(pos)] = phi
  state = set_ortho_center(state, [last(pos)])
  @assert isortho(state) && only(ortho_center(state)) == last(pos)
  normalize && (state[last(pos)] ./= norm(state[last(pos)]))
  # TODO: return maxtruncerr, will not be correct in cases where insertion executes multiple factorizations
  return state, spec
end

function insert_local_tensor(state::AbstractTTN, phi::ITensor, e::NamedEdge; kwargs...)
  state[dst(e)] *= phi
  state = set_ortho_center(state, [dst(e)])
  return state, nothing
end

#TODO: clean this up:
# also can we entirely rely on directionality of edges by construction?
current_ortho(::Type{<:Vector{<:V}}, st) where {V} = first(st)
current_ortho(::Type{NamedEdge{V}}, st) where {V} = src(st)
current_ortho(st) = current_ortho(typeof(st), st)

function region_update(
  projected_operator, state; outputlevel, which_sweep, sweep_plan, which_region_update
)
  (region, region_kwargs) = sweep_plan[which_region_update]
  (; extracter_kwargs, updater_kwargs, inserter_kwargs, internal_kwargs) = region_kwargs

  # these are equivalent to pop!(collection,key)
  (; extracter) = extracter_kwargs
  extracter_kwargs = Base.structdiff((; extracter), extracter_kwargs)
  (; updater) = updater_kwargs   #extract updater from  updater_kwargs
  updater_kwargs = Base.structdiff((; updater), updater_kwargs)
  (; inserter) = inserter_kwargs
  inserter_kwargs = Base.structdiff((; inserter), inserter_kwargs)

  region = first(sweep_plan[which_region_update])
  state, projected_operator, phi = extract_local_tensor(
    state, projected_operator, region; extracter_kwargs..., internal_kwargs
  )
  state! = Ref(state) # create references, in case solver does (out-of-place) modify PH or state
  projected_operator! = Ref(projected_operator)
  phi, info = updater(
    phi;
    state!,
    projected_operator!,
    outputlevel,
    which_sweep,
    sweep_plan,
    which_region_update,
    updater_kwargs,
    internal_kwargs,
  )  # args passed by reference are supposed to be modified out of place
  state = state![] # dereference
  projected_operator = projected_operator![]
  if !(phi isa ITensor && info isa NamedTuple)
    println("Solver returned the following types: $(typeof(phi)), $(typeof(info))")
    error("In alternating_update, solver must return an ITensor and a NamedTuple")
  end
  #haskey(region_kwargs,:normalize) && ( region_kwargs.normalize && (phi /= norm(phi)) )
  # ToDo: implement noise term as updater
  #drho = nothing
  #ortho = "left"    #i guess with respect to ordered vertices that's valid but may be cleaner to use next_region logic
  #if noise > 0.0 && isforward(direction)
  #  drho = noise * noiseterm(PH, phi, ortho) # TODO: actually implement this for trees...
  # so noiseterm is a solver
  #end

  state, spec = insert_local_tensor(state, phi, region; inserter_kwargs..., internal_kwargs)

  !haskey(region_kwargs, :region_printer) && (printer = default_region_update_printer)
  # only perform update! if region_observer actually passed as kwarg
  haskey(region_kwargs, :region_observer) && update!(
    region_observer!;
    cutoff,
    maxdim,
    mindim,
    which_region_update,
    sweep_plan,
    total_sweep_steps=length(sweep_plan),
    end_of_sweep=(which_region_update == length(sweep_plan)),
    state,
    region,
    which_sweep,
    spec,
    outputlevel,
    info...,
    region_kwargs...,
  )

  printer(;
    cutoff,
    maxdim,
    mindim,
    which_region_update,
    sweep_plan,
    total_sweep_steps=length(sweep_plan),
    end_of_sweep=(which_region_update == length(sweep_plan)),
    state,
    region,
    which_sweep,
    spec,
    outputlevel,
    info...,
    region_kwargs...,
  )
  return state, projected_operator
end
