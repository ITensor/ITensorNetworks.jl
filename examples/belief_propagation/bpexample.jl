using Compat
using ITensors
using Metis
using ITensorNetworks
using Random
using SplitApplyCombine
using NamedGraphs

using ITensorNetworks:
  belief_propagation, approx_network_region, contract_inner, message_tensors

function main()
  n = 4
  g_dims = (n, n)
  g = named_grid(g_dims)
  s = siteinds("S=1/2", g)
  chi = 2

  Random.seed!(5467)

  #bra
  ψ = randomITensorNetwork(s; link_space=chi)
  #bra-ket (but not actually contracted)
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])

  #Site to take expectation value on
  v = (1, 1)

  #Now do Simple Belief Propagation to Measure Sz on Site v
  pψψ = PartitionedGraph(ψψ, collect(values(group(v -> v[1], vertices(ψψ)))))
  mts = belief_propagation(
    pψψ; contract_kwargs=(; alg="exact"), verbose=true, niters=10, target_precision=1e-3
  )
  numerator_tensors = approx_network_region(
    pψψ, mts, [(v, 1)]; verts_tensors=[apply(op("Sz", s[v]), ψ[v])]
  )
  denominator_tensors = approx_network_region(pψψ, mts, [(v, 1)])
  sz_bp = contract(numerator_tensors)[] / contract(denominator_tensors)[]

  println(
    "Simple Belief Propagation Gives Sz on Site " * string(v) * " as " * string(sz_bp)
  )

  #Now do Column-wise General Belief Propagation to Measure Sz on Site v
  pψψ = PartitionedGraph(ψψ, collect(values(group(v -> v[1][1], vertices(ψψ)))))
  mts = belief_propagation(
    pψψ; contract_kwargs=(; alg="exact"), verbose=true, niters=10, target_precision=1e-3
  )
  numerator_tensors = approx_network_region(
    pψψ, mts, [(v, 1)]; verts_tensors=[apply(op("Sz", s[v]), ψ[v])]
  )
  denominator_tensors = approx_network_region(pψψ, mts, [(v, 1)])
  sz_gen_bp = contract(numerator_tensors)[] / contract(denominator_tensors)[]

  println(
    "General Belief Propagation Gives Sz on Site " * string(v) * " as " * string(sz_gen_bp)
  )

  #Now do General Belief Propagation with Matrix Product State Message Tensors Measure Sz on Site v
  ψψ = flatten_networks(ψ, dag(ψ); combine_linkinds=false, map_bra_linkinds=prime)
  Oψ = copy(ψ)
  Oψ[v] = apply(op("Sz", s[v]), ψ[v])
  ψOψ = flatten_networks(ψ, dag(Oψ); combine_linkinds=false, map_bra_linkinds=prime)

  combiners = linkinds_combiners(ψψ)
  ψψ = combine_linkinds(ψψ, combiners)
  ψOψ = combine_linkinds(ψOψ, combiners)

  pψψ = PartitionedGraph(ψψ, collect(values(group(v -> v[1], vertices(ψψ)))))
  mts = belief_propagation(
    pψψ;
    itensor_constructor=inds_e -> ITensor[dense(delta(i)) for i in inds_e],
    contract_kwargs=(;
      alg="density_matrix",
      output_structure=path_graph_structure,
      maxdim=8,
      contraction_sequence_alg="optimal",
    ),
  )
  numerator_tensors = approx_network_region(pψψ, mts, [v]; verts_tensors=[ψOψ[v]])
  denominator_tensors = approx_network_region(pψψ, mts, [v])
  sz_MPS_bp = contract(numerator_tensors)[] / contract(denominator_tensors)[]

  println(
    "Column-Wise MPS Belief Propagation Gives Sz on Site " *
    string(v) *
    " as " *
    string(sz_gen_bp),
  )

  #Now do it exactly
  sz_exact = contract(ψOψ)[] / contract(ψψ)[]

  return println("The exact value of Sz on Site " * string(v) * " is " * string(sz_exact))
end

main()
