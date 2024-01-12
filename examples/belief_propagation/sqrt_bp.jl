using ITensors
using ITensorNetworks
using Random
using SplitApplyCombine
using NamedGraphs

using ITensorNetworks:
  approx_network_region,
  belief_propagation,
  sqrt_belief_propagation,
  ising_network_state,
  message_tensors

function main(; n, niters, network="ising", β=nothing, h=nothing, χ=nothing)
  g_dims = (n, n)
  g = named_grid(g_dims)
  s = siteinds("S=1/2", g)

  Random.seed!(5467)

  ψ = if network == "ising"
    ising_network_state(s, β; h)
  elseif network == "random"
    randomITensorNetwork(s; link_space=χ)
  else
    error("Network type $network not supported.")
  end

  ψψ = norm_network(ψ)

  # Site to take expectation value on
  v = (n ÷ 2, n ÷ 2)

  #Now do Simple Belief Propagation to Measure Sz on Site v
  pψψ = PartitionedGraph(ψψ, collect(values(group(v -> v[1], vertices(ψψ)))))
  mts = message_tensors(pψψ)

  mts = @time belief_propagation(pψψ, mts; niters, contract_kwargs=(; alg="exact"))
  numerator_network = approx_network_region(
    pψψ, mts, [(v, 1)]; verts_tn=ITensor[apply(op("Sz", s[v]), ψ[v])])
  denominator_network = approx_network_region(pψψ, mts, [(v, 1)])
  sz_bp =
    ITensors.contract(
      numerator_network; sequence=contraction_sequence(numerator_network; alg="optimal")
    )[] / ITensors.contract(
      denominator_network; sequence=contraction_sequence(denominator_network; alg="optimal")
    )[]

  println(
    "Simple Belief Propagation Gives Sz on Site " * string(v) * " as " * string(sz_bp)
  )

  mts_sqrt = message_tensors(pψψ)
  mts_sqrt = @time sqrt_belief_propagation(pψψ, mts_sqrt; niters)

  numerator_network = approx_network_region(
    pψψ, mts_sqrt, [(v, 1)]; verts_tn=ITensor[apply(op("Sz", s[v]), ψ[v])])
  denominator_network = approx_network_region(pψψ, mts_sqrt, [(v, 1)])
  sz_sqrt_bp =
    contract(
      numerator_network; sequence=contraction_sequence(numerator_network; alg="optimal")
    )[] / contract(
      denominator_network; sequence=contraction_sequence(denominator_network; alg="optimal")
    )[]

  println(
    "Sqrt Belief Propagation Gives Sz on Site " * string(v) * " as " * string(sz_sqrt_bp)
  )

  return (; sz_bp, sz_sqrt_bp)
end
