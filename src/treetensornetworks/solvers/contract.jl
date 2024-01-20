function contract(
  ::Algorithm"fit",
  tn1::AbstractTTN,
  tn2::AbstractTTN;
  init=random_ttn(flatten_external_indsnetwork(tn1, tn2); link_space=trivial_space(tn1)),
  nsweeps=1,
  nsites=2, # used to be default of call to default_sweep_regions
  updater_kwargs=(;),
  kwargs...,
)
  n = nv(tn1)
  n != nv(tn2) && throw(
    DimensionMismatch("Number of sites operator ($n) and state ($(nv(tn2))) do not match")
  )
  if n == 1
    v = only(vertices(tn2))
    return typeof(tn2)([tn1[v] * tn2[v]])
  end

  # check_hascommoninds(siteinds, tn1, tn2)

  # In case `tn1` and `tn2` have the same internal indices
  tn1 = sim(linkinds, tn1)

  # In case `init` and `tn2` have the same internal indices
  init = sim(linkinds, init)

  # Fix site and link inds of init
  ## init = deepcopy(init)
  ## init = sim(linkinds, init)
  ## for v in vertices(tn2)
  ##   replaceinds!(
  ##     init[v], siteinds(init, v), uniqueinds(siteinds(tn1, v), siteinds(tn2, v))
  ##   )
  ## end

  PH = ProjTTNApply(tn2, tn1)
  sweep_plan = default_sweep_regions(nsites, init; kwargs...)
  psi = alternating_update(
    contract_updater, PH, init; nsweeps, sweep_plan, updater_kwargs, kwargs...
  )

  return psi
end

"""
Overload of `ITensors.contract`.
"""
function contract(tn1::AbstractTTN, tn2::AbstractTTN; alg="fit", kwargs...)
  return contract(Algorithm(alg), tn1, tn2; kwargs...)
end

"""
Overload of `ITensors.apply`.
"""
function apply(tn1::AbstractTTN, tn2::AbstractTTN; kwargs...)
  tn12 = contract(tn1, tn2; kwargs...)
  return replaceprime(tn12, 1 => 0)
end
