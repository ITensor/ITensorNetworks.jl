
function _extend_sweeps_param(param, nsweeps)
  if param isa Number
    eparam = fill(param, nsweeps)
  else
    length(param) >= nsweeps && return param[1:nsweeps]
    eparam = Vector(undef, nsweeps)
    eparam[1:length(param)] = param
    eparam[(length(param) + 1):end] .= param[end]
  end
  return eparam
end

function process_sweeps(
  nsweeps;
  cutoff=fill(1E-16, nsweeps),
  maxdim=fill(typemax(Int), nsweeps),
  mindim=fill(1, nsweeps),
  noise=fill(0.0, nsweeps),
  kwargs...,
)
  maxdim = _extend_sweeps_param(maxdim, nsweeps)
  mindim = _extend_sweeps_param(mindim, nsweeps)
  cutoff = _extend_sweeps_param(cutoff, nsweeps)
  noise = _extend_sweeps_param(noise, nsweeps)
  return maxdim, mindim, cutoff, noise, kwargs
end

function sweep_printer(; outputlevel, x, sweep, sw_time)
  if outputlevel >= 1
    print("After sweep ", sweep, ":")
    print(" maxlinkdim=", maxlinkdim(x))
    print(" cpu_time=", round(sw_time; digits=3))
    println()
    flush(stdout)
  end
end

function alternating_update(
  solver,
  problem_cache,
  x;
  checkdone=(; kws...) -> false,
  outputlevel::Integer=0,
  nsweeps::Integer=1,
  (sweep_observer!)=observer(),
  sweep_printer=sweep_printer,
  write_when_maxdim_exceeds::Union{Int,Nothing}=nothing,
  kwargs...,
)
  maxdim, mindim, cutoff, noise, kwargs = process_sweeps(nsweeps; kwargs...)
  insert_function!(sweep_observer!, "sweep_printer" => sweep_printer)

  for sweep in 1:nsweeps
    if !isnothing(write_when_maxdim_exceeds) && maxdim[sweep] > write_when_maxdim_exceeds
      if outputlevel >= 2
        println(
          "write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim[sweep] = $(maxdim[sweep]), writing environment tensors to disk",
        )
      end
      problem_cache = disk(problem_cache)
    end

    sw_time = @elapsed begin
      problem_cache = update_step(
        solver,
        problem_cache,
        x;
        outputlevel,
        sweep,
        maxdim=maxdim[sweep],
        mindim=mindim[sweep],
        cutoff=cutoff[sweep],
        noise=noise[sweep],
        kwargs...,
      )
    end

    update!(sweep_observer!; x, sweep, sw_time, outputlevel)

    checkdone(; x, sweep, outputlevel, kwargs...) && break
  end
  select!(sweep_observer!, Observers.DataFrames.Not("sweep_printer")) # remove sweep_printer
  return problem_cache
end

default_inds_map(x; kwargs...) = mapprime(x, 0 => 1; kwargs...)
default_inv_inds_map(x; kwargs...) = mapprime(x, 1 => 0; kwargs...)
default_contract_alg(x) = "bp"

abstract type AbstractITensorNetworkCache end
const AbstractITNCache = AbstractITensorNetworkCache

# TODO: Define `ITensorNetworkMap` that is an `ITensorNetwork` with an `inds_map`.
struct BPCache{TN,Cache,V,In,Out,Map} <: AbstractITNCache
  tn::TN # ITensorNetwork (unpartitioned? of quadratic form)
  cache::Cache # DataGraph of message tensors
  vs::Vector{V} # Vertices of original tensor network state
  update_region::Vector{V} # Region to update
  in_vs::In # Bra vertices of tensor network state
  out_vs::Out # Ket vertices of tensor network state
  inds_map::Map # Map indices from ket to bra
end

function set_nsite(cache::BPCache, nsite)
  error("Not implemented")
end

function position(cache::BPCache, nsite)
  error("Not implemented")
end

function cache(contract_alg::Algorithm"bp", tn::AbstractITensorNetwork, vs::Vector, in_vs::Function, out_vs::Function, inds_map::Function)
  return BPCache(tn, DataGraph(), vs, in_vs, out_vs, inds_map)
end

function cache(tn::AbstractITensorNetwork, vs::Vector, in_vs::Function, out_vs::Function, inds_map::Function; contract_alg)
  return cache(Algorithm(contract_alg), tn, vs, in_vs, out_vs, inds_map)
end

# ⟨x|A|x⟩ / ⟨x|x⟩
struct RayleighQuotientCache{Num<:AbstractITNCache,Den<:AbstractITNCache} <: AbstractITNCache
  num::Num
  den::Den
end

function set_nsite(cache::RayleighQuotientCache, nsite)
  return RayleighQuotientCache(set_nsite(cache.num, nsite), set_nsite(cache.den, nsite))
end

# Rayleigh quotient numerator network, ⟨x|A|x⟩
# Also known as a [quadratic form](https://en.wikipedia.org/wiki/Quadratic_form).
# TODO: Allow customizing vertex map.
function quadratic_form_network(
  A::AbstractITensorNetwork,
  x::AbstractITensorNetwork;
  inds_map=default_inds_map,
)
  xAx = ⊗(x, A, inds_map(dag(x)))
  return xAx
end

function quadratic_form_network(x::AbstractITensorNetwork; inds_map=default_inds_map)
  xx = quadratic_form_network(id_network(x; inds_map), x; inds_map)
  return xx
end

function id_network(inds::AbstractIndsNetwork; inds_map=default_inds_map)
  id_net = ITensorNetwork(vertices(inds))
  for v in vertices(inds)
    setindex_preserve_graph!(id_net, ITensor(true), v)
    for i in inds[v]
      setindex_preserve_graph!(id_net, id_net[v] * δ(dag(i), i'), v)
    end
  end
  return id_net
end

function id_network(x::AbstractITensorNetwork; inds_map=default_inds_map)
  return id_network(siteinds(x))
end

# Rayleigh quotient numerator cache, ⟨x|A|x⟩
# Also known as a [quadratic form](https://en.wikipedia.org/wiki/Quadratic_form).
function quadratic_form_cache(
  A::AbstractITensorNetwork,
  x::AbstractITensorNetwork;
  inds_map=default_inds_map,
  inv_inds_map=default_inv_inds_map,
  contract_alg=default_contract_alg(x),
)
  vs = vertices(x)
  xAx = quadratic_form_network(A, x; inds_map)
  xAx_in_vs(v) = (v, 1)
  xAx_out_vs(v) = (v, 3)
  xAx_inds_map = inds_map ∘ dag
  xAx_cache = cache(xAx, vs, xAx_in_vs, xAx_out_vs, xAx_inds_map; contract_alg)
  return xAx_cache
end

# Rayleigh quotient denominator cache, ⟨x|x⟩
# https://en.wikipedia.org/wiki/Norm_(mathematics)
# https://en.wikipedia.org/wiki/Inner_product_space
function quadratic_form_cache(
  x::AbstractITensorNetwork;
  inds_map=default_inds_map,
  kwargs...,
)
  return quadratic_form_cache(id_network(x; inds_map), x; inds_map, kwargs...)
end

# Cache for a tensor network representation of a
# [Rayleigh quotient](https://en.wikipedia.org/wiki/Rayleigh_quotient).
# TODO: Detect if there is an orthogonality center and if so
# avoid making the denominator cache of the Rayleigh quotient.
function rayleigh_quotient_cache(
  A::AbstractITensorNetwork,
  x::AbstractITensorNetwork;
  kwargs...,
)
  xAx_cache = quadratic_form_cache(A, x; kwargs...)
  xx_cache = quadratic_form_cache(x; kwargs...)
  return RayleighQuotientCache(xAx_cache, xx_cache)
end

function alternating_update(solver, A::AbstractITensorNetwork, x₀::AbstractITensorNetwork; kwargs...)
  xAx_xx_cache = rayleigh_quotient_cache(A, x₀)
  return alternating_update(solver, xAx_xx_cache, x₀; kwargs...)
end

## function alternating_update(solver, A::AbstractVector{<:AbstractITensorNetwork}, x₀::AbstractITensorNetwork; kwargs...)
##   return alternating_update(solver, Sum(A), x₀; kwargs...)
## end
## 
## function alternating_update(solver, A::Sum, x₀::AbstractITensorNetwork; kwargs...)
##   A_cache = rayleigh_quotient_cache(A, x₀)
##   return alternating_update(solver, A_cache, x₀; kwargs...)
## end
