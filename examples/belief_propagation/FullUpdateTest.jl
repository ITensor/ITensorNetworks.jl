using ITensors
using ITensors: optimal_contraction_sequence
using ITensorVisualizationBase
using ITensorNetworks
using Statistics
using ITensorNetworks:
  orthogonalize,
  norm_sqr,
  construct_initial_mts,
  get_environment,
  rename_vertices_itn,
  calculate_contraction,
  group_partition_vertices,
  compute_message_tensors
using KaHyPar
using Dictionaries
using Compat
using Random
using LinearAlgebra
using NPZ
using NamedGraphs
using SplitApplyCombine

function RandomUnitaryMatrix(N::Int)
  x = (rand(N, N) + rand(N, N) * im) / sqrt(2)
  f = qr(x)
  diagR = sign.(real(diag(f.R)))
  diagR[diagR .== 0] .= 1
  diagRm = diagm(diagR)
  u = f.Q * diagRm

  return u
end

function ITensors.op(::OpName"RandomU", ::SiteType"S=1/2")
  return RandomUnitaryMatrix(4)
end

function ITensors.op(::OpName"RandomNonU", ::SiteType"S=1/2")
  return randn(4, 4)
end

function form_mts(ψ::ITensorNetwork, niters::Int64, nsites::Int64)
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])
  vertex_groups = group(v -> v[1], vertices(ψψ))
  Z_p = partition(partition(ψψ, vertex_groups); nvertices_per_partition=nsites)
  Z_verts = [
    reduce(vcat, (vertices(Z_p[vp][v]) for v in vertices(Z_p[vp]))) for vp in vertices(Z_p)
  ]
  Z_p = partition(ψψ, Z_verts)

  mts = construct_initial_mts(ψψ, Z_p; init=(I...) -> @compat allequal(I) ? 1 : 0)
  mts = update_all_mts(ψψ, mts, niters)

  return mts, ψψ
end

function generate_random_U1_state(ngates::Int64, s::IndsNetwork, chi::Int64, dims)
  ψ = ITensorNetwork(s, v -> isodd(dims[2] * (v[1] - 1) + v[2]) ? "↑" : "↓")

  for i in 1:ngates
    e = edges(ψ)[rand((1, length(edges(ψ))))]
    qubits_to_act_on = [src(e), dst(e)]
    s1, s2 = s[qubits_to_act_on[1]], s[qubits_to_act_on[2]]
    hj =
      4 * op("Sz", s1) * op("Sz", s2) - 2 * op("S+", s1) * op("S-", s2) -
      2 * op("S-", s1) * op("S+", s2) + 0.2 * op("Sz", s1) * op("I", s2) -
      0.2 * 0.2 * op("Sz", s2) * op("I", s1)
    gate = exp(-3 * randn() * hj)

    ψ = ITensorNetworks.orthogonalize(ψ, qubits_to_act_on[1])
    ψψ = ψ ⊗ prime(dag(ψ); sites=[])
    vertex_groups = group_partition_vertices(ψψ, v -> v[1]; nvertices_per_partition=1)
    mts = compute_message_tensors(ψψ; vertex_groups=vertex_groups, niters=0)
    ψ[qubits_to_act_on[1]], ψ[qubits_to_act_on[2]], f = apply_gate(
      ψψ, gate, mts, qubits_to_act_on[1], qubits_to_act_on[2]; nsweeps=0, maxdim=χ
    )
  end

  return ψ
end

function test_gate_application(
  ngates::Int64,
  s::IndsNetwork,
  χ::Int64,
  nsites::Int64,
  nsweeps::Int64,
  dims;
  conserve=false,
  unitary=true,
)
  delta = 1.2
  fSUs = Float64[]
  fBPs = Float64[]
  for i in 1:ngates
    if (conserve)
      ψ = generate_random_U1_state(100, s, χ, dims)
    else
      ψ = randomITensorNetwork(s; link_space=χ)
    end
    e = edges(ψ)[rand((1, length(edges(ψ))))]
    qubits_to_act_on = [src(e), dst(e)]
    s1, s2 = s[qubits_to_act_on[1]], s[qubits_to_act_on[2]]
    if (conserve)
      hj =
        4 * delta * op("Sz", s1) * op("Sz", s2) - 2 * op("S+", s1) * op("S-", s2) -
        2 * op("S-", s1) * op("S+", s2) + 0.2 * op("Sz", s1) * op("I", s2) -
        0.2 * 0.2 * op("Sz", s2) * op("I", s1)
      gate = exp(-3 * randn() * hj)
    else
      if (unitary)
        gate = ITensors.op("RandomU", s1..., s2...)
      else
        gate = ITensors.op("RandomNonU", s1..., s2...)
      end
    end
    ψ = ITensorNetworks.orthogonalize(ψ, qubits_to_act_on[1])
    ψactual = apply(gate, ψ; maxdim=4 * χ, ortho=false, normalize=true)

    ψψ = ψ ⊗ prime(dag(ψ); sites=[])
    vertex_groups = group_partition_vertices(ψψ, v -> v[1]; nvertices_per_partition=nsites)
    mts = compute_message_tensors(ψψ; vertex_groups=vertex_groups)
    ψ[qubits_to_act_on[1]], ψ[qubits_to_act_on[2]], f = apply_gate(
      ψψ,
      gate,
      mts,
      qubits_to_act_on[1],
      qubits_to_act_on[2];
      nsweeps=nsweeps,
      maxdim=χ,
      fidelity_loss=true,
    )
    fBP =
      contract_inner(ψ, ψactual) /
      (sqrt(contract_inner(ψactual, ψactual)) * sqrt(contract_inner(ψ, ψ)))
    ψ[qubits_to_act_on[1]], ψ[qubits_to_act_on[2]], f = apply_gate(
      ψψ,
      gate,
      mts,
      qubits_to_act_on[1],
      qubits_to_act_on[2];
      nsweeps=0,
      maxdim=χ,
      fidelity_loss=true,
    )
    fSU =
      contract_inner(ψ, ψactual) /
      (sqrt(contract_inner(ψactual, ψactual)) * sqrt(contract_inner(ψ, ψ)))

    push!(fSUs, fSU * conj(fSU))
    push!(fBPs, fBP * conj(fBP))
  end

  @show mean(fBPs)
  @show mean(fSUs)

  @show -(1.0 / length(fBPs)) * log(prod(fBPs))
  @show -(1.0 / length(fSUs)) * log(prod(fSUs))
end

include("FullUpdate.jl")

ITensors.disable_warn_order()

Random.seed!(1234)

dims = (5, 5)
n = prod(dims)
g = named_grid(dims)
conserve = false
unitary = false
s = siteinds("S=1/2", g; conserve_qns=conserve)

χ = 3

no_gates = 500
nsites, nsweeps = 1, 10
test_gate_application(
  no_gates, s, χ, nsites, nsweeps, dims; conserve=conserve, unitary=unitary
)
