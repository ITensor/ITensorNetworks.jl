using Compat
using ITensors
using Metis
using ITensorNetworks
using Random
using NamedGraphs: add_vertex!, add_edge!
using AbstractTrees
using CairoMakie
using Permutations
using Statistics

using ITensorNetworks:
  compute_message_tensors,
  calculate_contraction,
  group_partition_vertices,
  contract_inner,
  split_index,
  ising_network,
  get_environment,
  find_subgraph,
  update_all_mts

using OMEinsumContractionOrders
include("ContractionTreeFunctions.jl")

function calculate_Z(contract_tree, tn::ITensorNetwork, mts::DataGraph, vertex_names)
  contract_edges = contraction_edges(contract_tree)
  ds = []
  for ce in contract_edges
    TL, TR = edge_bipartition(contract_tree, ce)
    envs = vcat(
      get_environment(tn, mts, vertex_names[TL]), get_environment(tn, mts, vertex_names[TR])
    )
    push!(
      ds, ITensors.contract(envs; sequence=ITensors.optimal_contraction_sequence(envs))[]
    )
  end

  ns = []
  for nv in internal_contraction_nodes(contract_tree)
    T1, T2, T3 = vertex_names[nv[1]], vertex_names[nv[2]], vertex_names[nv[3]]
    envs = vcat(
      get_environment(tn, mts, T1; dir=:out),
      get_environment(tn, mts, T2; dir=:out),
      get_environment(tn, mts, T3; dir=:out),
    )
    push!(
      ns, ITensors.contract(envs; sequence=ITensors.optimal_contraction_sequence(envs))[]
    )
  end

  for nv in external_contraction_nodes(contract_tree)
    int_Ts = external_contraction_node_int_tensors(contract_tree, nv)
    ext_Ts = external_contraction_node_ext_tensors(contract_tree, nv)
    envs = get_environment(tn, mts, vertex_names[ext_Ts]; dir=:out)
    tensors_to_contract = vcat(envs, ITensor[tn[vertex_names[i]] for i in int_Ts])
    push!(
      ns,
      ITensors.contract(
        tensors_to_contract;
        sequence=ITensors.optimal_contraction_sequence(tensors_to_contract),
      )[],
    )
  end

  return prod(ns) / prod(ds)
end

function compute_Z(tn::ITensorNetwork, nvertices_per_partition::Int64, seq)
  mts = compute_message_tensors(tn; nvertices_per_partition=nvertices_per_partition)

  g = create_contraction_tree(seq)
  vertex_names = [v for v in vertices(tn)]
  return calculate_Z(g, tn, mts, vertex_names)
end

function reconstruct_state(tn::ITensorNetwork; nvertices_per_partition=1)
  state_vecs = Iterators.product(ntuple(i -> 0:1, length(vertices(tn)))...)
  vs = vertices(tn)
  actual_amplitudes = []
  approx_amplitudes = []

  seq_state_vec = zeros(length(vertices(tn)))
  tn_red_seq = copy(tn)
  count = 1
  for v in vs
    siteind = inds(tn_red_seq[v]; tags="Site")
    delta_tensor = ITensor([1 - seq_state_vec[count], seq_state_vec[count]], siteind)
    tn_red_seq[v] = tn_red_seq[v] * delta_tensor
    count += 1
  end
  seq = contraction_sequence([tn_red_seq[v] for v in vertices(tn_red_seq)]; alg="optimal")

  for state_vec in state_vecs
    @show state_vec
    tn_red = copy(tn)
    count = 1
    for v in vs
      siteind = inds(tn_red[v]; tags="Site")
      delta_tensor = ITensor([1 - state_vec[count], state_vec[count]], siteind)
      tn_red[v] = tn_red[v] * delta_tensor
    end
    count += 1

    push!(approx_amplitudes, compute_Z(tn_red, nvertices_per_partition, seq))
    push!(actual_amplitudes, ITensors.contract(tn_red)[])
  end

  normalize!(actual_amplitudes)
  normalize!(approx_amplitudes)

  f = dot(actual_amplitudes, approx_amplitudes)

  return f * conj(f)
end

function state_reconstruction(g, namplitudes::Int64; nvertices_per_partition=1, bond_dim=2)
  s = siteinds("S=1/2", g)
  fids = []
  for i in 1:namplitudes
    ψ = randomITensorNetwork(s; link_space=bond_dim)
    push!(fids, reconstruct_state(ψ; nvertices_per_partition=nvertices_per_partition))
  end

  return fids
end

n = 4
dims = (n, n)
g = named_grid(dims)
s = siteinds("S=1/2", g)
Random.seed!(1234)
namplitudes = 1
nvertices_per_partition = 1
bond_dim = 2
ψ = randomITensorNetwork(s; link_space=bond_dim)
ψψ = flatten_network(ψ, ψ)

seq = contract_sequence(ψψ)
@show seq
