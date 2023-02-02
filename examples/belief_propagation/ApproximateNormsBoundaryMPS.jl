using Compat
using ITensors
using Metis
using ITensorNetworks
using Random
using Random: bitrand
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
  update_all_mts,
  contract_boundary_mps

#GIVEN AN ITENSOR NETWORK CORRESPONDING to an Lx*Ly grid with sites indexed as (i,j) then perform contraction using a sequence of mps-mpo contractions
function contract_boundary_mps_n(tn::ITensorNetwork, centre_column::Int64; kwargs...)
  dims = maximum(vertices(tn))
  d1, d2 = dims
  vL = MPS([tn[i1, 1] for i1 in 1:d1])
  for i2 in 2:(centre_column - 1)
    T = MPO([tn[i1, i2] for i1 in 1:d1])
    vL = contract(T, vL; kwargs...)
  end
  vR = MPS([tn[i1, d2] for i1 in 1:d1])
  for i2 in (d2 - 1):-1:(centre_column + 1)
    T = MPO([tn[i1, i2] for i1 in 1:d1])
    vR = contract(T, vR; kwargs...)
  end
  T = MPO([tn[i1, centre_column] for i1 in 1:d1])
  return inner(dag(vL), T, vR)[]
end

#GIVEN AN ITENSOR NETWORK CORRESPONDING to an Lx*Ly grid with sites indexed as (i,j) then perform contraction using a sequence of mps-mpo contractions
function contract_boundary_mps_d(tn::ITensorNetwork, left_column::Int64; kwargs...)
  dims = maximum(vertices(tn))
  d1, d2 = dims
  vL = MPS([tn[i1, 1] for i1 in 1:d1])
  for i2 in 2:(left_column)
    T = MPO([tn[i1, i2] for i1 in 1:d1])
    vL = contract(T, vL; kwargs...)
  end
  vR = MPS([tn[i1, d2] for i1 in 1:d1])
  for i2 in (d2 - 1):-1:(left_column + 1)
    T = MPO([tn[i1, i2] for i1 in 1:d1])
    vR = contract(T, vR; kwargs...)
  end
  return inner(dag(vL), vR)[]
end

function contract_boundary_mps_full(tn::ITensorNetwork; kwargs...)
  dims = maximum(vertices(tn))
  d1, d2 = dims
  ns, ds = [], []
  for i in 2:(d2 - 1)
    push!(ns, contract_boundary_mps_n(tn, i; kwargs...))
    if (i != d2 - 1)
      push!(ds, contract_boundary_mps_d(tn, i; kwargs...))
    end
  end
  return prod(ns) / prod(ds)
end

function reconstruct_state(tn::ITensorNetwork, Boundary_bond_dim::Int64)
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

  no_state_vecs = 100
  for count in 1:no_state_vecs
    state_vec = bitrand(length(vertices(tn)))
    tn_red = copy(tn)
    count = 1
    for v in vs
      siteind = inds(tn_red[v]; tags="Site")
      delta_tensor = ITensor([1 - state_vec[count], state_vec[count]], siteind)
      tn_red[v] = tn_red[v] * delta_tensor
    end

    push!(approx_amplitudes, contract_boundary_mps_full(tn_red; maxdim=Boundary_bond_dim))
    push!(actual_amplitudes, contract_boundary_mps(tn_red))
  end

  normalize!(actual_amplitudes)
  normalize!(approx_amplitudes)

  f = dot(actual_amplitudes, approx_amplitudes)

  return f * conj(f)
end

function state_reconstruction(
  g, namplitudes::Int64, bond_dim::Int64, Boundary_bond_dim::Int64
)
  s = siteinds("S=1/2", g)
  fids = []
  for i in 1:namplitudes
    ψ = randomITensorNetwork(s; link_space=bond_dim)
    push!(fids, reconstruct_state(ψ, Boundary_bond_dim))
    @show fids
  end

  return fids
end

#n =7
dims = (9, 6)
g = named_grid(dims)
s = siteinds("S=1/2", g)
Random.seed!(1234)
bond_dim = 5
Boundary_bond_dim = 2

namplitudes = 10
fids = state_reconstruction(g, namplitudes, bond_dim, Boundary_bond_dim)

@show fids
