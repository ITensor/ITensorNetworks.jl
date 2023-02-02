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
  calculate_contraction
using KaHyPar
using Dictionaries
using Compat
using Random
using LinearAlgebra
using NPZ
using NamedGraphs
using SplitApplyCombine
using CairoMakie

maybe_only(x) = x
maybe_only(x::Tuple{T}) where {T} = only(x)

function tebd_XXZ(
  ψ::AbstractITensorNetwork,
  s::IndsNetwork,
  conserve;
  Delta,
  dbetas,
  χ=2,
  print_frequency=10,
  niters=0,
  nsites=1,
  nsweeps=5,
  actual_state=nothing,
)
  ψ = copy(ψ)
  fidelities = Float64[]
  per_gate_fidelities = Float64[]
  steps_taken = Float64[]
  nsteps = length(dbetas)
  Z_actual = sqrt((actual_state * dag(actual_state))[])
  for step in 1:nsteps
    if step % print_frequency == 0
      @show step
      if (actual_state != nothing)
        approx_state_vec = ITensors.contract(
          [ψ[v] for v in vertices(ψ)];
          sequence=ITensors.optimal_contraction_sequence([ψ[v] for v in vertices(ψ)]),
        )
        f =
          (actual_state * dag(approx_state_vec))[] /
          (sqrt((approx_state_vec * dag(approx_state_vec))[]) * Z_actual)
        println("Actual Fidelity is Currently " * string(f * conj(f)))
        push!(steps_taken, step - 1)
        push!(fidelities, f * conj(f))
      end
    end
    gates = XXZ_gates(s; Delta, Δβ=dbetas[step], rev=true)
    for gate in gates
      qubits_to_act_on = vertices(s)[findall(
        i -> (length(commoninds(s[i], inds(gate))) != 0), vertices(s)
      )]
      if (length(qubits_to_act_on) == 1)
        ψ[qubits_to_act_on[1]] = noprime!(ψ[qubits_to_act_on[1]] * gate)
      else
        ψ = ITensorNetworks.orthogonalize(ψ, qubits_to_act_on[1])
        ψψ = ψ ⊗ prime(dag(ψ); sites=[])
        if (nsites == 2)
          v1 = [
            (qubits_to_act_on[1], 1),
            (qubits_to_act_on[1], 2),
            (qubits_to_act_on[2], 1),
            (qubits_to_act_on[2], 2),
          ]
          complement_vertices = [v for v in vertices(ψψ) if v ∉ v1]
          vertex_groups = vcat(
            group_partition_vertices(
              underlying_graph(ψψ)[complement_vertices],
              v -> v[1];
              nvertices_per_partition=2,
            ),
            [v1],
          )
        else
          vertex_groups = group_partition_vertices(
            ψψ, v -> v[1]; nvertices_per_partition=nsites
          )
        end
        mts = compute_message_tensors(ψψ; vertex_groups=vertex_groups)

        #ψexact=  copy(ψ)
        #ψexact[qubits_to_act_on[1]], ψexact[qubits_to_act_on[2]], f = apply_gate(ψψ, gate, mts, qubits_to_act_on[1], qubits_to_act_on[2]; nsweeps = 0, maxdim = 4*χ,fidelity_loss=true)
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

        #f = contract_inner(ψ, ψexact)[]/sqrt(contract_inner(ψ, ψ)[]*contract_inner(ψexact, ψexact)[])
        #push!(per_gate_fidelities, f*conj(f))
        push!(per_gate_fidelities, 0)
      end
    end
  end

  return ψ, fidelities, steps_taken, per_gate_fidelities
end

function XXZ_gates(s::IndsNetwork; Delta=1.0, Δβ=0.1, rev=true)
  gates = ITensor[]
  for e in edges(s)
    hj =
      -4 * Delta * op("Sz", s[maybe_only(src(e))]) * op("Sz", s[maybe_only(dst(e))]) -
      2 * op("S+", s[maybe_only(src(e))]) * op("S-", s[maybe_only(dst(e))]) -
      2 * op("S-", s[maybe_only(src(e))]) * op("S+", s[maybe_only(dst(e))])
    Gj = exp(-Δβ * 0.5 * hj)
    push!(gates, Gj)
  end

  if (rev)
    append!(gates, reverse(gates))
  end
  return gates
end

function XXZ_4th_order_gates(s::IndsNetwork; Delta=1.0, Δβ=0.1, rev=true)
  gates = ITensor[]
  p = 0.41449077
  for e in edges(s)
    hj =
      4 * Delta * op("Sz", s[maybe_only(src(e))]) * op("Sz", s[maybe_only(dst(e))]) -
      2 * op("S+", s[maybe_only(src(e))]) * op("S-", s[maybe_only(dst(e))]) -
      2 * op("S-", s[maybe_only(src(e))]) * op("S+", s[maybe_only(dst(e))])
    Gj = exp(-Δβ * 0.5 * p * hj)
    push!(gates, Gj)
  end

  append!(gates, reverse(gates))
  append!(gates, gates)
  return gates
end

function XXZ_energy(ψ::ITensorNetwork, s::IndsNetwork, Delta::Float64, bond_dim::Int64)
  gates = XXZ_gates(s; Delta=Delta, reverse=false)
  seq = contraction_sequence(inner_network(ψ, ψ; combine_linkinds=true))
  Z = contract_inner(ψ, ψ; sequence=seq)
  e = 0
  for gate in gates
    Oψ = apply(gate, ψ; maxdim=4 * bond_dim)
    e += contract_inner(ψ, Oψ; sequence=seq) / Z
  end

  return e
end

function XXZ_energy_BP(
  ψ::ITensorNetwork, s::IndsNetwork, Delta::Float64, niters::Int64, nsites::Int64
)
  mts, ψψ = form_mts(ψ, niters, nsites)
  gates = XXZ_gates(s; Delta=Delta)
  e = 0
  for gate in gates
    qubits_to_act_on = vertices(s)[findall(
      i -> (length(commoninds(s[i], inds(gate))) != 0), vertices(s)
    )]
    num = calculate_contraction(
      ψψ,
      mts,
      [(q, 1) for q in qubits_to_act_on],
      [noprime!(prod([ψ[q] for q in qubits_to_act_on]) * gate)],
    )[]
    denom = calculate_contraction(
      ψψ, mts, [(q, 1) for q in qubits_to_act_on], [prod([ψ[q] for q in qubits_to_act_on])]
    )[]
    e += num / denom
  end
  return e
end

function create_adj_mat(g::NamedGraph)
  n_sites = length(vertices(g))
  adj_mat = zeros((n_sites, n_sites))
  name_map = Dict{Int64,Tuple}()
  verts = vertices(g)
  count = 1
  for v in verts
    for vn in neighbors(g, v)
      index = findfirst(x -> x == vn, verts)
      adj_mat[count, index] = 1
    end
    name_map[count] = v
    count += 1
  end
  return adj_mat, name_map
end

function XXZ_DMRG_backend(dims, chi_max, delta, adj_mat, name_map, s::IndsNetwork)
  L = prod(dims)
  sites = siteinds("S=1/2", L; conserve_qns=true)
  init_state = [isodd(i) ? "Up" : "Dn" for i in 1:L]

  sweeps = Sweeps(10)
  setmaxdim!(
    sweeps,
    trunc(Int, chi_max / 8),
    trunc(Int, chi_max / 4),
    trunc(Int, chi_max / 2),
    trunc(Int, chi_max),
  )
  setcutoff!(sweeps, 1E-10)

  ampo = OpSum()
  for i in 1:L
    for j in (i + 1):L
      ampo += 4 * adj_mat[i, j] * delta, "Sz", i, "Sz", j
      ampo += -2 * adj_mat[i, j], "S+", i, "S-", j
      ampo += -2 * adj_mat[i, j], "S-", i, "S+", j
    end
  end
  H = MPO(ampo, sites)

  psi0 = randomMPS(sites, init_state)
  e, psi = dmrg(H, psi0, sweeps)

  for i in 1:L
    sind = siteind(psi, i)
    replaceind!(psi[i], sind, s[name_map[i]])
  end

  println("DMRG Finished with an e of " * string(e))

  return psi
end

include("FullUpdate.jl")

Random.seed!(1234)

ITensors.disable_warn_order()

dims = (4, 4)
n = prod(dims)
g = named_grid(dims)
adj_mat, name_map = create_adj_mat(g)

s = siteinds("S=1/2", g; conserve_qns=true)

Delta = 0.75

ψ_GS = XXZ_DMRG_backend(dims, 100, Delta, adj_mat, name_map, s)
ψ_GS_tn = ITensorNetwork([ψ_GS[j] for j in eachindex(ψ_GS)])
ψ_GS_tn = rename_vertices_itn(ψ_GS_tn, Dictionary(vertices(ψ_GS_tn), values(name_map)))
ψ_GS_tn = ITensors.contract(
  [ψ_GS_tn[v] for v in vertices(ψ_GS_tn)];
  sequence=ITensors.optimal_contraction_sequence([ψ_GS_tn[v] for v in vertices(ψ_GS_tn)]),
)

χ = 3

ψ_init = ITensorNetwork(s, v -> isodd(dims[2] * (v[2] - 1) + v[1]) ? "↑" : "↓")

dbetas = vcat([0.1 for i in 1:40])

ψ_SU, fids_SU, steps_taken_SU, per_gate_f_SU = tebd_XXZ(
  ψ_init,
  s,
  conserve;
  Delta,
  dbetas,
  χ=χ,
  print_frequency=1,
  niters=0,
  nsweeps=0,
  actual_state=ψ_GS_tn,
)
#ψ_SBP, fids_SBP, steps_taken_SBP, per_gate_f_SBP = tebd_XXZ(ψ_init, s, conserve; Delta, dbetas, χ=χ, print_frequency = 1,niters = 10, nsites =1, actual_state = ψ_GS_tn)
ψ_SBP, fids_SBP, steps_taken_SBP, per_gate_f_SBP = tebd_XXZ(
  ψ_init,
  s,
  conserve;
  Delta,
  dbetas,
  χ=χ,
  print_frequency=1,
  niters=10,
  nsites=1,
  actual_state=ψ_GS_tn,
)

lw = 4
fig = Figure(; resolution=(600, 600))
colors = ["black", "red", "blue"]
ax = Axis(fig[1, 1]; xlabel="Steps Taken", ylabel="Fidelity")
#lines!(ax, steps_taken_GBP, fids_GBP, label = "GBP (Two Site)", linewidth = lw, color = colors[3])
lines!(ax, steps_taken_SBP, fids_SBP; label="SBP", linewidth=lw, color=colors[1])
lines!(ax, steps_taken_SU, fids_SU; label="SU", linewidth=lw, color=colors[2])

leg = Legend(fig[1, 2], ax)

save(
  "/mnt/home/jtindall/Documents/RoughFigures/ImaginaryTimeEvolution/MyPlotLx" *
  string(dims[1]) *
  "BD" *
  string(χ) *
  ".pdf",
  fig,
)

fig = Figure(; resolution=(600, 600))
ax = Axis(fig[1, 1]; xlabel="Steps Taken", ylabel="Fidelity")
lines!(
  ax,
  [i for i in 1:length(per_gate_f_SBP)],
  (per_gate_f_SBP - per_gate_f_SU);
  linewidth=lw,
  color=colors[1],
)

save(
  "/mnt/home/jtindall/Documents/RoughFigures/ImaginaryTimeEvolution/GateByGateLx" *
  string(dims[1]) *
  "BD" *
  string(χ) *
  ".pdf",
  fig,
)
