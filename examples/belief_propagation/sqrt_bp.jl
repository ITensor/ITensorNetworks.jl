using ITensors
using ITensorNetworks
using Random
using SplitApplyCombine

using ITensorNetworks:
  approx_network_region,
  belief_propagation,
  sqrt_belief_propagation,
  ising_network_state,
  message_tensors

function main(;
  n,
  niters,
  network="ising",
  β=nothing,
  h=nothing,
  χ=nothing,
)
  g_dims = (n, n)
  @show g_dims
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
  @show v

  #Now do Simple Belief Propagation to Measure Sz on Site v
  mts = message_tensors(
    ψψ; subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ))))
  )

  mts = @time belief_propagation(ψψ, mts; niters, contract_kwargs=(; alg="exact"))
  numerator_network = approx_network_region(
    ψψ, mts, [(v, 1)]; verts_tn=ITensorNetwork([apply(op("Sz", s[v]), ψ[v])])
  )
  denominator_network = approx_network_region(ψψ, mts, [(v, 1)])
  sz_bp = 2 * contract(numerator_network; sequence=contraction_sequence(numerator_network; alg="optimal"))[] / contract(denominator_network; sequence=contraction_sequence(denominator_network; alg="optimal"))[]

  println(
    "Simple Belief Propagation Gives Sz on Site " * string(v) * " as " * string(sz_bp)
  )

  mts_sqrt = message_tensors(
    ψψ; subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ))))
  )

  mts_sqrt = @time sqrt_belief_propagation(ψ, mts_sqrt; niters)

  numerator_network = approx_network_region(
    ψψ, mts_sqrt, [(v, 1)]; verts_tn=ITensorNetwork([apply(op("Sz", s[v]), ψ[v])])
  )
  denominator_network = approx_network_region(ψψ, mts_sqrt, [(v, 1)])
  sz_bp = 2 * contract(numerator_network; sequence=contraction_sequence(numerator_network; alg="optimal"))[] / contract(denominator_network; sequence=contraction_sequence(denominator_network; alg="optimal"))[]

  println(
    "Sqrt Belief Propagation Gives Sz on Site " * string(v) * " as " * string(sz_bp)
  )

  return (; mts, mts_sqrt)
end
