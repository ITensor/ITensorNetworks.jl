using ITensors
using ITensorVisualizationBase
using ITensorNetworks
using Statistics
using NPZ
using NamedGraphs: add_vertex!
using Dictionaries
using Compat
using Random

include("FullUpdate.jl")

function ITensors.op(::OpName"RootX", ::SiteType"S=1/2")
  return (1 / sqrt(2)) * [
    1 -im
    -im 1
  ]
end

function ITensors.op(::OpName"RootY", ::SiteType"S=1/2")
  return (1 / sqrt(2)) * [
    1 -1
    1 1
  ]
end

function ITensors.op(::OpName"RootW", ::SiteType"S=1/2")
  return (1 / sqrt(2)) * [
    1 -sqrt(im)
    sqrt(-im) 1
  ]
end

function ITensors.op(::OpName"fSim", ::SiteType"S=1/2")
  return [
    1 0 0 0
    0 0 -im 0
    0 -im 0 0
    0 0 0 exp(-im * pi / 6)
  ]
end

function sycamore_graph(nhoriz::Int64, nvert::Int64; delete_rogue_vertex=true)
  g = NamedGraph([(i, j) for i in 1:nhoriz for j in 1:nvert])
  for v in vertices(g)
    x, y = v[1], v[2]
    if (isodd(y))
      if (has_vertex(g, (x, y + 1)))
        add_edge!(g, (x, y) => (x, y + 1))
      end

      if (has_vertex(g, (x - 1, y + 1)))
        add_edge!(g, (x, y) => (x - 1, y + 1))
      end
    else
      if (has_vertex(g, (x + 1, y + 1)))
        add_edge!(g, (x, y) => (x + 1, y + 1))
      end

      if (has_vertex(g, (x, y + 1)))
        add_edge!(g, (x, y) => (x, y + 1))
      end
    end
  end

  if (has_vertex(g, (4, 9)) && delete_rogue_vertex == true)
    rem_vertex!(g, (4, 9))
  end
  return g
end

function sycamore_graph_edge_group(sycamore_g::NamedGraph, seq)
  es = edges(sycamore_g)
  es_out = []
  for e in es
    x1, y1, x2, y2 = src(e)[1], src(e)[2], dst(e)[1], dst(e)[2]
    if (y1 > y2)
      x1, y1, x2, y2 = dst(e)[1], dst(e)[2], src(e)[1], src(e)[2]
    end
    if (x2 == x1 && y2 == y1 + 1 && isodd(y1) && seq == "A")
      push!(es_out, e)
    elseif (x2 == x1 + 1 && y2 == y1 + 1 && iseven(y1) && seq == "B")
      push!(es_out, e)
    elseif (x2 == x1 && y2 == y1 + 1 && iseven(y1) && seq == "C")
      push!(es_out, e)
    elseif (x2 == x1 - 1 && y2 == y1 + 1 && isodd(y1) && seq == "D")
      push!(es_out, e)
    end
  end
  return es_out
end

function sycamore_circuit(
  sycamore_g::NamedGraph, nhoriz::Int64, nvert::Int64, seq::String; input_dict=nothing
)
  gates = []
  single_particle_dict = Dict{Tuple,String}()

  for v in vertices(sycamore_g)
    string_set = ["RootX", "RootY", "RootW"]
    if (input_dict != nothing)
      deleteat!(string_set, string_set .== input_dict[v])
    end
    ind = rand(1:length(string_set))
    push!(gates, (string_set[ind], [v]))
    single_particle_dict[v] = string_set[ind]
  end
  es = sycamore_graph_edge_group(sycamore_g, seq)
  for e in es
    push!(gates, ("fSim", [src(e), dst(e)]))
  end

  return gates, single_particle_dict
end

function sycamore_test_simulation(no_cycles, nvert, nhoriz, max_dim)
  sycamore_g = sycamore_graph(nhoriz, nvert)
  s = siteinds("S=1/2", sycamore_g)

  cycle_sequence = ["A", "B", "C", "D", "C", "D", "A", "B"]
  seq_counter = 1
  single_part_gates_used = Dict{Tuple,String}([v => "I" for v in vertices(sycamore_g)])
  bond_dims = DataGraph{vertextype(sycamore_g),Int64,Int64}(sycamore_g)
  for e in edges(sycamore_g)
    bond_dims[e] = 1
  end

  for i in 1:no_cycles
    seq = cycle_sequence[seq_counter]

    gates, single_part_gates_used = sycamore_circuit(
      sycamore_g, nhoriz, nvert, seq; input_dict=single_part_gates_used
    )

    for gate in gates
      if (length(gate[2]) > 1)
        q1, q2 = gate[2][1], gate[2][2]
        e = edgetype(sycamore_g)(q1 => q2)
        q1_neighbours = neighbors(sycamore_g, q1)
        prod_chiL = 1
        for q1n in q1_neighbours
          if (q1n != q2)
            prod_chiL = prod_chiL * bond_dims[edgetype(sycamore_g)(q1 => q1n)]
          end
        end

        q2_neighbours = neighbors(sycamore_g, q2)
        prod_chiR = 1
        for q2n in q2_neighbours
          if (q2n != q1)
            prod_chiR = prod_chiR * bond_dims[edgetype(sycamore_g)(q2 => q2n)]
          end
        end

        chiLtilde = min(2 * bond_dims[e], prod_chiL)
        chiRtilde = min(2 * bond_dims[e], prod_chiR)
        bond_dims[e] = min(2 * min(chiLtilde, chiRtilde), max_dim)
      end
    end

    seq_counter = (seq_counter % length(cycle_sequence)) + 1
  end

  memory_count = 0
  for v in vertices(sycamore_g)
    no_coeffs = 2
    for vn in neighbors(sycamore_g, v)
      no_coeffs *= bond_dims[edgetype(sycamore_g)(v => vn)]
    end
    memory_count += 16 * no_coeffs
  end

  println(
    "Memory Requirement for Exact is " * string(memory_count / (1000000000)) * " Gigabytes"
  )

  return bond_dims
end

nvert = 9
nhoriz = 6

no_cycles = 20
max_dim = 100
bond_dims = sycamore_test_simulation(no_cycles, nvert, nhoriz, max_dim)
