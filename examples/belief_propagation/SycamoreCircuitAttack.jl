using ITensors
using ITensors: optimal_contraction_sequence
using ITensorVisualizationBase
using ITensorNetworks
using Statistics
using ITensorNetworks: orthogonalize
using KaHyPar
using Dictionaries
using Compat
using Random
using LinearAlgebra

include("beliefpropagationV2.jl")

function ITensors.op(::OpName"RootX", ::SiteType"S=1/2")
  return [
    0.5+0.5 * im 0.5-0.5 * im
    0.5-0.5 * im 0.5+0.5 * im
  ]
end

function ITensors.op(::OpName"RootY", ::SiteType"S=1/2")
  return [
    0.5+0.5 * im -0.5-0.5 * im
    0.5+0.5 * im 0.5+0.5 * im
  ]
end

function ITensors.op(::OpName"RootW", ::SiteType"S=1/2")
  return [
    0.5+0.5 * im -im/sqrt(2)
    1/sqrt(2) 0.5+0.5 * im
  ]
end

function ITensors.op(::OpName"Rz", ::SiteType"S=1/2"; phi=0)
  return [
    exp(-im * phi / 2) 0
    0 exp(im * phi / 2)
  ]
end

function ITensors.op(::OpName"FSim", ::SiteType"S=1/2"; theta=0, phi=0)
  return [
    1 0 0 0
    0 cos(theta) -im*sin(theta) 0
    0 -im*sin(theta) cos(theta) 0
    0 0 0 exp(-im * phi)
  ]
end

function get_op(gate, s::IndsNetwork)
  qubits_involved = gate[length(gate)]
  if (length(qubits_involved) == 2)
    return op(
      gate[1],
      s[qubits_involved[1]]...,
      s[qubits_involved[2]]...;
      theta=gate[2][1],
      phi=gate[2][2],
    )
  else
    if (length(gate) == 2)
      return op(gate[1], s[qubits_involved[1]])
    else
      return op(gate[1], s[qubits_involved[1]]; phi=gate[2])
    end
  end
end

function form_mts(ψ::ITensorNetwork, niters::Int64, nsites::Int64)
  mts = construct_initial_mts_V2(
    ψ, nsites; flattened=false, init=(I...) -> @compat allequal(I) ? 1 : 0
  )
  mts = update_all_mts_V2(ψ, mts, niters; flattened=false)

  return mts
end

function simulate_sycamore(
  nqubits::Int64,
  ncycles::Int64,
  seed::Int64,
  section::String,
  mode::String,
  seq::String;
  bond_dim=2,
)
  if (nqubits < 53)
    file_str1 =
      "/mnt/home/jtindall/Documents/Data/SycamoreData/NPZCircuitData/circuit_n53_m20_s" *
      string(seed) *
      "_e0_pABCDCDAB.npz"
    file_str2 =
      "/mnt/home/jtindall/Documents/Data/SycamoreData/NPZCircuitData/circuit_n" *
      string(nqubits) *
      "_m14_s0_e0_pEFGH.npz"
    _, _, gates = load_circuit(file_str1)
    qubits, edges, _ = load_circuit(file_str2)
  else
    file_str =
      "/mnt/home/jtindall/Documents/Data/SycamoreData/NPZCircuitData/circuit_n53_m20_s" *
      string(seed) *
      "_e0_pABCDCDAB.npz"
    qubits, edges, gates = load_circuit(file_str)
  end

  g_syc = build_sycamore_graph(qubits, edges, section)
  s = siteinds("S=1/2", g_syc)
  gates_skipped = 0
  cycle_no = 1
  actual_cycle_no = 1
  if (length(vertices(g_syc)) <= 28)
    compute_actual = true
  else
    println("Too Many Qubits to Do anything Exactly")
    compute_actual = false
  end

  if (compute_actual)
    init = (I...) -> allequal(I) && I[1] != 1 ? 1 : 0
    psi_actual = itensor(
      [
        init(Tuple(I)...) for
        I in CartesianIndices(tuple(dim.([s[v][] for v in vertices(s)])...))
      ],
      [s[v][] for v in vertices(s)],
    )
  end

  f_cur_cycle = 1
  fidelities = zeros(ncycles)
  cycle_fs = zeros(ncycles)
  fs_gen = Float64[]
  psi_approx = ITensorNetwork(s, v -> "↓")
  niters, nsites, nsweeps = 10, 1, 10
  for gate in gates
    qubits_to_act_on = gate[length(gate)]
    skip_gate = false
    for qubit in qubits_to_act_on
      if (qubit ∉ vertices(g_syc))
        skip_gate = true
      end
    end
    if (skip_gate == false)
      it_gate = get_op(gate, s)
      if (
        (cycle_no % 2 == 1 && gate[1] == "Rz") ||
        (cycle_no % 2 == 0 && length(qubits_to_act_on) == 1 && gate[1] != "Rz")
      )
        cycle_no += 1
        if (cycle_no % 2 == 1)
          if (compute_actual)
            z_actual = (psi_actual * dag(psi_actual))[]
            z_approx = contract_inner(psi_approx, psi_approx)
            f =
              ITensors.contract(
                vcat([psi_approx[v] for v in vertices(psi_approx)], dag(psi_actual))
              )[] / sqrt(z_actual * z_approx)
            fidelities[actual_cycle_no] = f * conj(f)
          end

          cycle_fs[actual_cycle_no] = f_cur_cycle
          f_cur_cycle = 1
          actual_cycle_no += 1
        end
      end
      if (actual_cycle_no > ncycles)
        println("Now on Cycle " * string(actual_cycle_no) * " so exiting.")
        break
      end

      if (length(qubits_to_act_on) == 1)
        psi_approx[qubits_to_act_on[1]] = noprime!(
          psi_approx[qubits_to_act_on[1]] * it_gate
        )
      else
        if (mode == "BP")
          psi_approx = ITensorNetworks.orthogonalize(psi_approx, qubits_to_act_on[1])
          mts = form_mts(psi_approx, niters, nsites)
          #psi_approx = apply_gate(psi_approx, it_gate, mts, qubits_to_act_on[1], qubits_to_act_on[2], nsweeps; maxdim= bond_dim)
          psi_approx, fid = apply_gate_SBP(
            psi_approx,
            it_gate,
            mts,
            qubits_to_act_on[1],
            qubits_to_act_on[2],
            nsweeps;
            maxdim=bond_dim,
            fidelity_loss=true,
          )
          push!(fs_gen, fid)
          f_cur_cycle *= fid
        else
          psi_approx = apply(it_gate, psi_approx; cutoff=1e-10, maxdim=bond_dim, ortho=true)
        end
      end

      if (compute_actual)
        psi_actual = psi_actual * it_gate
        noprime!(psi_actual)
      end
    else
      gates_skipped += 1
    end
  end

  return fidelities, fs_gen, cycle_fs
end

include("FullUpdate.jl")
include("SycamoreFunctions.jl")

nqubits = 53
ncycles = 20
seed = 0
no_elided = 0
seq = "ABCDCDAB"
bond_dim = 1
ITensors.disable_warn_order()

#sinds = [s[v][] for v in vertices(s)]
#C = combiner(sinds)
#final_state_vector = vector(psi_final*C)

fidelities, fs_gen, cycle_fs = simulate_sycamore(
  nqubits, ncycles, seed, "F", "BP", seq; bond_dim=bond_dim
)

@show fidelities
@show fs_gen
@show cycle_fs
#exact_R_st, s_R = simulate_sycamore(nqubits, ncycles, seed, "R", "Exact", seq)
#exact_F_st, s_F = simulate_sycamore(nqubits, ncycles, seed, "F", "Exact", seq)
