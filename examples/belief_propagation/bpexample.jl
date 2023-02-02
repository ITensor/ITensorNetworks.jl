using Compat
using ITensors
using Metis
using ITensorNetworks
using Random
using SplitApplyCombine

using ITensorNetworks:
  compute_message_tensors, calculate_contraction, contract_inner, nested_graph_leaf_vertices

function main()
  n = 4
  dims = (n, n)
  g = named_grid(dims)
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
  nsites = 1

  vertex_groups = nested_graph_leaf_vertices(
    partition(partition(ψψ, group(v -> v[1], vertices(ψψ))); nvertices_per_partition=nsites)
  )
  mts = compute_message_tensors(ψψ; vertex_groups=vertex_groups)
  sz_bp =
    calculate_contraction(
      ψψ, mts, [(v, 1)]; verts_tensors=ITensor[apply(op("Sz", s[v]), ψ[v])]
    )[] / calculate_contraction(ψψ, mts, [(v, 1)])[]

  println(
    "Simple Belief Propagation Gives Sz on Site " * string(v) * " as " * string(sz_bp)
  )

  #Now do General Belief Propagation to Measure Sz on Site v
  nsites = 4
  vertex_groups = nested_graph_leaf_vertices(
    partition(partition(ψψ, group(v -> v[1], vertices(ψψ))); nvertices_per_partition=nsites)
  )
  mts = compute_message_tensors(ψψ; vertex_groups=vertex_groups)
  sz_bp =
    calculate_contraction(
      ψψ, mts, [(v, 1)]; verts_tensors=ITensor[apply(op("Sz", s[v]), ψ[v])]
    )[] / calculate_contraction(ψψ, mts, [(v, 1)])[]

  println(
    "General Belief Propagation (2-site subgraphs) Gives Sz on Site " *
    string(v) *
    " as " *
    string(sz_bp),
  )

  #Now do it exactly
  Oψ = copy(ψ)
  Oψ[v] = apply(op("Sz", s[v]), ψ[v])
  sz_exact = contract_inner(Oψ, ψ) / contract_inner(ψ, ψ)

  return println("The exact value of Sz on Site " * string(v) * " is " * string(sz_exact))
end

main()
