using ITensors
using ITensors: optimal_contraction_sequence
using ITensorVisualizationBase
using ITensorNetworks
using Statistics
using ITensorNetworks: orthogonalize, compute_message_tensors, group_partition_vertices
using KaHyPar
using Dictionaries
using Compat
using Random
using LinearAlgebra
using NPZ

include("FullUpdate.jl")
include("SycamoreFunctions.jl")

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

function take_overlap(psi_approx::ITensorNetwork, psi_exact::ITensor)
  tensors_to_contract = ITensor[psi_approx[v] for v in vertices(psi_approx)]
  psi_approx_sv = ITensors.contract(
    tensors_to_contract; sequence=ITensors.optimal_contraction_sequence(tensors_to_contract)
  )
  C = combiner(inds(psi_approx_sv))
  return dot(normalize(array(psi_approx_sv * C)), normalize(array(psi_exact * C)))
end

function calculate_max_next_bond_dimension(psi::ITensorNetwork, q1::Tuple, q2::Tuple)
  q1_neighbours = neighbors(psi, q1)
  prod_chiL = 1
  for q1n in q1_neighbours
    if (q1n != q2)
      prod_chiL = prod_chiL * dim(commoninds(psi[q1n], psi[q1]))
    end
  end

  q2_neighbours = neighbors(psi, q2)
  prod_chiR = 1
  for q2n in q2_neighbours
    if (q2n != q1)
      prod_chiR = prod_chiR * dim(commoninds(psi[q2n], psi[q2]))
    end
  end

  chiLtilde = min(2 * dim(commoninds(psi[q1], psi[q2])), prod_chiL)
  chiRtilde = min(2 * dim(commoninds(psi[q1], psi[q2])), prod_chiR)
  return 2 * min(chiLtilde, chiRtilde)
end

function simulate_sycamore(
  nqubits::Int64, seed::Int64, section::String, no_cycles::Int64, mode::String; χ=2
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
  compute_actual = length(vertices(g_syc)) <= 28 ? true : false

  if (compute_actual)
    init = (I...) -> allequal(I) && I[1] != 1 ? 1 : 0
    ψ_actual = itensor(
      [
        init(Tuple(I)...) for
        I in CartesianIndices(tuple(dim.([s[v][] for v in vertices(s)])...))
      ],
      [s[v][] for v in vertices(s)],
    )
  end

  f_cur_cycle = 1
  fidelities = zeros(no_cycles)
  cycle_fs = zeros(no_cycles)
  fs_gen = Float64[]
  ψ = ITensorNetwork(s, v -> "↓")
  niters, nsites, nsweeps = 10, 1, 10
  count = 1
  for gate in gates
    qubits_to_act_on = gate[length(gate)]
    skip_gate = false
    for qubit in qubits_to_act_on
      if (qubit ∉ vertices(g_syc))
        skip_gate = true
      end
    end
    if (!skip_gate)
      it_gate = get_op(gate, s)
      if (
        (cycle_no % 2 == 1 && gate[1] == "Rz") ||
        (cycle_no % 2 == 0 && length(qubits_to_act_on) == 1 && gate[1] != "Rz")
      )
        cycle_no += 1
        if (cycle_no % 2 == 1)
          if (compute_actual)
            f = take_overlap(ψ, ψ_actual)
            fidelities[actual_cycle_no] = f * conj(f)
          end

          cycle_fs[actual_cycle_no] = f_cur_cycle
          f_cur_cycle = 1
          actual_cycle_no += 1
        end
      end
      if (actual_cycle_no > no_cycles)
        println("Now on Cycle " * string(actual_cycle_no) * " so exiting.")
        break
      end

      if (length(qubits_to_act_on) == 1)
        ψ[qubits_to_act_on[1]] = noprime!(ψ[qubits_to_act_on[1]] * it_gate)
      else
        println("On 2QBT gate " * string(count))
        flush(stdout)
        count += 1
        max_dim = calculate_max_next_bond_dimension(
          ψ, qubits_to_act_on[1], qubits_to_act_on[2]
        )
        if (χ >= max_dim || mode != "BP")
          niters, nsweeps = 0, 0
        else
          niters, nsweeps = 10, 5
          psi_approx = ITensorNetworks.orthogonalize(ψ, qubits_to_act_on[1])
        end
        ψψ = ψ ⊗ prime(dag(ψ); sites=[])
        vertex_groups = group_partition_vertices(ψψ, v -> v[1]; nvertices_per_partition=1)
        mts = compute_message_tensors(ψψ; vertex_groups=vertex_groups)
        ψ[qubits_to_act_on[1]], ψ[qubits_to_act_on[2]], f = apply_gate(
          ψψ,
          get_op(gate, s),
          mts,
          qubits_to_act_on[1],
          qubits_to_act_on[2];
          nsweeps=nsweeps,
          maxdim=χ,
          fidelity_loss=true,
        )
        push!(fs_gen, f)
        f_cur_cycle *= f
      end

      if (compute_actual)
        ψ_actual = ψ_actual * it_gate
        noprime!(ψ_actual)
      end
    else
      gates_skipped += 1
    end
  end

  return fidelities, fs_gen, cycle_fs
end

# nqubits = parse(Int64, ARGS[1])
# bond_dim = parse(Int64, ARGS[2])
# no_cycles = parse(Int64, ARGS[3])

nqubits = 20
χ = 2
no_cycles = 20

seed = 0
no_elided = 0
section = "F"
mode = "BP"
ITensors.disable_warn_order()

println(
  "Simulating " *
  string(nqubits) *
  " qubits. Input seed is " *
  string(seed) *
  ". Section is " *
  section *
  ". Mode is " *
  mode *
  ". No Cycles is " *
  string(no_cycles),
)

fidelities, fs_gen, cycle_fs = simulate_sycamore(
  nqubits, seed, section, no_cycles, mode; χ=χ
)

file_str =
  "/mnt/ceph/users/jtindall/Data/SycamoreData/Fidelities/" *
  mode *
  "/Mode" *
  mode *
  "BondDim" *
  string(χ) *
  "Sec" *
  section *
  "Circuit_n" *
  string(nqubits) *
  "_m" *
  string(no_cycles) *
  "_s" *
  string(seed) *
  "_e" *
  string(no_elided) *
  "_pABCDCDAB.npz"

#npzwrite(file_str, fidelities = fidelities, fs_gen = fs_gen, cycle_fs=cycle_fs)

@show fidelities
