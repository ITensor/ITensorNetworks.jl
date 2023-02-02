using NamedGraphs: add_edge!
using PyCall

np = pyimport("numpy")

function load_circuit(file_name::String)
  data = np.load(file_name; allow_pickle=true)
  qubits = get(data, "qubits")
  gates = get(data, "gates")
  edges = get(data, "edges")

  act_edges = []
  for i in 1:size(edges)[1]
    push!(act_edges, (edges[i, 1, 1], edges[i, 1, 2]) => (edges[i, 2, 1], edges[i, 2, 2]))
  end

  act_qubits = []
  for i in 1:size(qubits)[1]
    push!(act_qubits, (qubits[i, 1], qubits[i, 2]))
  end

  act_gates = []
  for gate in gates
    push!(act_gates, convert(PyAny, gate))
  end

  return act_qubits, act_edges, act_gates
end

function build_sycamore_graph(qubits, edges, section::String)
  g = NamedGraph(qubits)
  for e in edges
    add_edge!(g, e)
  end

  if (section == "F")
    return g
  elseif (section == "L")
    gL = deepcopy(g)
    for v in vertices(g)
      if (v[2] - v[1] >= 1)
        rem_vertex!(gL, v)
      end
    end

    return gL
  elseif (section == "R")
    gR = deepcopy(g)
    for v in vertices(g)
      if (v[2] - v[1] < 1)
        rem_vertex!(gR, v)
      end
    end

    return gR
  end
end
