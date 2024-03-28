function sum_contract(
  ::Algorithm"fit",
  tns::Vector{<:Tuple{<:AbstractTTN,<:AbstractTTN}};
  init,
  nsites=2,
  nsweeps=1,
  cutoff=eps(),
  updater=contract_updater,
  kwargs...,
)
  tn1s = first.(tns)
  tn2s = last.(tns)
  ns = nv.(tn1s)
  n = first(ns)
  any(ns .!= nv.(tn2s)) && throw(
    DimensionMismatch("Number of sites operator ($n) and state ($(nv(tn2))) do not match")
  )
  any(ns .!= n) &&
    throw(DimensionMismatch("Number of sites in different operators ($n) do not match"))
  # ToDo: Write test for single-vertex TTN, this implementation has not been tested.
  if n == 1
    res = 0
    for (tn1, tn2) in zip(tn1s, tn2s)
      v = only(vertices(tn2))
      res += tn1[v] * tn2[v]
    end
    return typeof(tn2)([res])
  end

  # In case `tn1` and `tn2` have the same internal indices
  operator = ProjOuterProdTTN{vertextype(first(tn1s))}[]
  for (tn1, tn2) in zip(tn1s, tn2s)
    tn1 = sim(linkinds, tn1)

    # In case `init` and `tn2` have the same internal indices
    init = sim(linkinds, init)
    push!(operator, ProjOuterProdTTN(tn2, tn1))
  end
  operator = isone(length(operator)) ? only(operator) : ProjTTNSum(operator)
  #ToDo: remove?
  # Fix site and link inds of init
  ## init = deepcopy(init)
  ## init = sim(linkinds, init)
  ## for v in vertices(tn2)
  ##   replaceinds!(
  ##     init[v], siteinds(init, v), uniqueinds(siteinds(tn1, v), siteinds(tn2, v))
  ##   )
  ## end

  return alternating_update(operator, init; nsweeps, nsites, updater, cutoff, kwargs...)
end

function NDTensors.contract(
  a::Algorithm"fit", tn1::AbstractTTN, tn2::AbstractTTN; kwargs...
)
  return sum_contract(a, [(tn1, tn2)]; kwargs...)
end

"""
Overload of `ITensors.contract`.
"""
function NDTensors.contract(tn1::AbstractTTN, tn2::AbstractTTN; alg="fit", kwargs...)
  return contract(Algorithm(alg), tn1, tn2; kwargs...)
end

"""
Overload of `ITensors.apply`.
"""
function apply(tn1::AbstractTTN, tn2::AbstractTTN; init, kwargs...)
  if !isone(plev_diff(flatten_external_indsnetwork(tn1, tn2), external_indsnetwork(init)))
    error(
      "Initial guess `init` needs to primelevel one less than the contraction tn1 and tn2."
    )
  end
  init = init'
  tn12 = contract(tn1, tn2; init, kwargs...)
  return replaceprime(tn12, 1 => 0)
end

function sum_apply(
  tns::Vector{<:Tuple{<:AbstractTTN,<:AbstractTTN}}; alg="fit", init, kwargs...
)
  if !isone(
    plev_diff(
      flatten_external_indsnetwork(first(first(tns)), last(first(tns))),
      external_indsnetwork(init),
    ),
  )
    error(
      "Initial guess `init` needs to primelevel one less than the contraction tn1 and tn2."
    )
  end

  init = init'
  tn12 = sum_contract(Algorithm(alg), tns; init, kwargs...)
  return replaceprime(tn12, 1 => 0)
end

function plev_diff(a::IndsNetwork, b::IndsNetwork)
  pla = plev(only(a[first(vertices(a))]))
  plb = plev(only(b[first(vertices(b))]))
  return pla - plb
end
