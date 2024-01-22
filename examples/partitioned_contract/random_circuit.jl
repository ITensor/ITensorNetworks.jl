using ITensors, Graphs, NamedGraphs, TimerOutputs
using Random
using PastaQ
using KaHyPar
using ITensorNetworks

# Return all the tensors represented by `gates`, and the `hilbert`
# after applying these gates. The `hilbert` is a vector if indices.
function gate_tensors(hilbert::Vector{<:Index}, gates; applydag=false)
  hilbert = copy(hilbert)
  Us = ITensor[]
  for g in gates
    if applydag == true
      t = dag(gate(hilbert, g))
      inds1 = [hilbert[n] for n in g[2]]
      t = swapinds(t, inds1, [i' for i in inds1])
    else
      t = gate(hilbert, g)
    end
    if length(g[2]) == 2
      ind = hilbert[g[2][1]]
      L, R = factorize(t, ind, ind'; cutoff=1e-16)
      push!(Us, L, R)
    else
      push!(Us, t)
    end
    for n in g[2]
      hilbert[n] = hilbert[n]'
    end
  end
  # @info "gates", gates
  # @info "hilbert", hilbert
  return Us, hilbert
end

function line_partition_gates(N, gates; order=1)
  @assert order in [1, 2, 3, 0]
  nrow, ncol = N
  if order in [0, 1]
    # each partition contains a column
    partition_qubits = [[(c - 1) * nrow + r for r in 1:nrow] for c in 1:ncol]
  elseif order == 2
    # Add columns [1], [2, 3], [4, 5], ...
    partition_qubits = [collect(1:nrow)]
    push!(
      partition_qubits,
      [
        [(c1 - 1) * nrow + r for r in 1:nrow for c1 in [c, c + 1]] for c in 2:2:(ncol - 1)
      ]...,
    )
    if iseven(ncol)
      push!(partition_qubits, [(ncol - 1) * nrow + r for r in 1:nrow])
    end
  else
    # Add columns [1, 2], [3, 4], [5, 6], ...
    partition_qubits = [
      [(c1 - 1) * nrow + r for r in 1:nrow for c1 in [c, c + 1]] for c in 1:2:ncol
    ]
    if isodd(ncol)
      push!(partition_qubits, [(ncol - 1) * nrow + r for r in 1:nrow])
    end
  end
  @info partition_qubits
  partition_gates = [filter(g -> all(q -> q in p, g[2]), gates) for p in partition_qubits]
  return partition_gates
end

function line_network(network::Vector)
  tntree = network[1]
  for i in 2:length(network)
    if network[i] isa Vector
      tntree = [tntree, network[i]]
    else
      tntree = [tntree, [network[i]]]
    end
  end
  return tntree
end

function random_circuit_line_partition_sequence(
  N, depth; twoqubitgates="CX", onequbitgates="Ry"
)
  @assert N isa Number || length(N) <= 2
  # Each layer contains multiple two-qubit gates, followed by one-qubit
  # gates, each applying on one qubit.
  layers = randomcircuit(
    N; depth=depth, twoqubitgates=twoqubitgates, onequbitgates=onequbitgates
  )
  partition_gates = []
  for (i, l) in enumerate(layers)
    push!(partition_gates, line_partition_gates(N, l; order=i % 4)...)
  end
  hilbert = qubits(prod(N))
  tensor_partitions = [productstate(hilbert)[:]]
  for gates in partition_gates
    tensors, hilbert = gate_tensors(hilbert, gates)
    push!(tensor_partitions, tensors)
  end
  for gates in reverse(partition_gates)
    tensors, hilbert = gate_tensors(hilbert, reverse(gates); applydag=true)
    push!(tensor_partitions, tensors)
  end
  push!(tensor_partitions, productstate(hilbert)[:])
  return line_network(tensor_partitions)
end
