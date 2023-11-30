using ITensors
using ITensorNetworks
using NamedGraphs
using NamedGraphs: rem_edges!, add_edge!, decorate_graph_edges, hexagonal_lattice_graph
using Graphs

using ITensorNetworks:
  initialize_bond_tensors,
  vidal_itn_canonicalness,
  vidal_to_symmetric_gauge,
  vidal_gauge,
  approx_network_region,
  optimal_contraction_sequence,
  contract,
  symmetric_to_vidal_gauge,
  norm_network

function heavy_hex_lattice_graph(n::Int64, m::Int64)
  g = hexagonal_lattice_graph(n, m)
  g = decorate_graph_edges(g)
  return g
end

function ibm_processor_graph(n::Int64, m::Int64)
  g = heavy_hex_lattice_graph(n, m)
  dims = maximum(vertices(hexagonal_lattice_graph(n, m)))
  v1, v2 = (dims[1], 1), (1, dims[2])
  add_vertices!(g, [v1, v2])
  add_edge!(g, v1 => v1 .- (1, 0))
  add_edge!(g, v2 => v2 .+ (1, 0))

  return g
end

eagle_processor_graph() = ibm_processor_graph(3, 6)
hummingbird_processor_graph() = ibm_processor_graph(2, 4)
osprey_processor_graph() = ibm_processor_graph(6, 12)

"""Take the expectation value of o on an ITN using belief propagation"""
function expect_state_SBP(
  o::ITensor, ψ::AbstractITensorNetwork, ψψ::AbstractITensorNetwork, mts::DataGraph
)
  Oψ = apply(o, ψ; cutoff=1e-16)
  ψ = copy(ψ)
  s = siteinds(ψ)
  vs = vertices(s)[findall(i -> (length(commoninds(s[i], inds(o))) != 0), vertices(s))]
  vs_braket = [(v, 1) for v in vs]

  numerator_network = approx_network_region(
    ψψ, mts, vs_braket; verts_tn=ITensorNetwork(ITensor[Oψ[v] for v in vs])
  )
  denominator_network = approx_network_region(ψψ, mts, vs_braket)
  num_seq = contraction_sequence(numerator_network; alg="optimal")
  den_seq = contraction_sequence(numerator_network; alg="optimal")
  return contract(numerator_network; sequence=num_seq)[] /
         contract(denominator_network; sequence=den_seq)[]
end

function main(θh::Float64, no_trotter_steps::Int64; apply_kwargs...)

  #Build the graph
  g = eagle_processor_graph()

  #Do this if measuring a Z based expectation value (i.e. ignore ZZ_gates in final layer as they are irrelevant)
  shortened_final_layer = true

  s = siteinds("S=1/2", g)

  #Gauging parameters
  target_c = 1e-3
  gauge_freq = 1

  #State initialisation
  ψ = ITensorNetwork(s, v -> "↑")
  bond_tensors = initialize_bond_tensors(ψ)

  #Build gates
  HX, HZZ = ising(g; h=θh / 2, J1=0), ising(g; h=0, J1=-pi / 4)
  RX, RZZ = exp(-im * HX; alg=Trotter{1}()), exp(-im * HZZ; alg=Trotter{1}())
  RX_gates, RZZ_gates = Vector{ITensor}(RX, s), Vector{ITensor}(RZZ, s)

  for i in 1:no_trotter_steps
    println("On Trotter Step $i")
    gates = if (shortened_final_layer && i == no_trotter_steps)
      RX_gates
    else
      vcat(RX_gates, RZZ_gates)
    end

    for gate in gates
      ψ, bond_tensors = apply(gate, ψ, bond_tensors; normalize=true, apply_kwargs...)
    end

    cur_C = vidal_itn_canonicalness(ψ, bond_tensors)
    if ((i + 1) % gauge_freq) == 0 && (cur_C >= target_c)
      println("Too far from the Vidal gauge. Regauging the state!")
      ψ, _ = vidal_to_symmetric_gauge(ψ, bond_tensors)
      ψ, bond_tensors = vidal_gauge(
        ψ; target_canonicalness=target_c, niters=20, cutoff=1e-14
      )
    end
  end

  #Calculate on-site magnetisations
  ψ, mts = vidal_to_symmetric_gauge(ψ, bond_tensors)
  ψψ = norm_network(ψ)
  mag_dict = Dict(
    zip(
      [v for v in vertices(ψ)],
      [expect_state_SBP(op("Z", s[v]), ψ, ψψ, mts) for v in vertices(ψ)],
    ),
  )

  return mag_dict
end

ibm_processor_graph(3, 6)

θh = pi / 4
no_trotter_steps = 5
χ = 32
apply_kwargs = (; maxdim=χ, cutoff=1e-12)

println(
  "Simulating $no_trotter_steps Steps of the Kicked Ising Model on the Heavy Hex Layout."
)
println("θh is $θh. Maxdim is $χ.")

mags = main(θh, no_trotter_steps; apply_kwargs...)

println(
  "After $no_trotter_steps steps. Average magnetisation is " *
  string(sum(values(mags)) / length(values(mags))),
)
