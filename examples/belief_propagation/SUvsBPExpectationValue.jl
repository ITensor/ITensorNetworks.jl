using ITensors
using ITensorNetworks
using Random
using Statistics
using NPZ
using ITensorNetworks:
  contract_boundary_mps,
  construct_initial_mts,
  update_all_mts,
  iterate_two_site_expec,
  orthogonalize,
  two_site_rdm_bp
using Compat
using LinearAlgebra
using Plots
using KaHyPar

function approx_two_site_rdm(psi::ITensorNetwork, v1::Tuple, v2::Tuple, s::IndsNetwork)
  if (!has_edge(psi, v1 => v2))
    println("ERROR: Graph does not posssess this edge")
  else
    psi_L = psi[v1] * psi[v2]
    psi_R = prime(dag(psi_L); tags="Site")
    rdm = psi_L * psi_R
    C = combiner(s[v1], s[v2])
    Cp = combiner(s[v1]', s[v2]')
    rdm = rdm * C * Cp
  end
  tr_rdm = rdm * delta(combinedind(C), combinedind(Cp))
  return array(rdm / tr_rdm)
end

function exact_two_site_rdm(ψ::ITensorNetwork, v1::Tuple, v2::Tuple, s::IndsNetwork)
  tn1 = sim(ψ; sites=[])
  tn2 = deepcopy(ψ)
  tn = ⊗(dag(tn1), tn2)
  for v in vertices(ψ)
    if (v != v1 && v != v2)
      tn = contract(tn, (v, 2) => (v, 1))
    end
  end

  prime!(tn[(v1, 2)]; tags="Site")
  prime!(tn[(v2, 2)]; tags="Site")

  rdm = ITensors.contract(tn)
  C = combiner(s[v1], s[v2])
  Cp = combiner(s[v1]', s[v2]')
  rdm = rdm * C * Cp
  tr_rdm = rdm * delta(combinedind(C), combinedind(Cp))
  return array(rdm / tr_rdm)
end

function trace_dist(A, B)
  evals, evecs = eigen(A - B)
  return 0.5 * sum(abs.(evals))
end

function benchmark(lx, ly)
  g = named_grid((lx, ly))
  chi = 2
  chi_max = 2
  s = siteinds("S=1/2", g)
  niters = 10

  no_dis = 100
  trace_dist_SUs = zeros(no_dis)
  trace_dist_ortho_SUs = zeros(no_dis)
  trace_dist_SBPs = zeros(no_dis)
  trace_dist_GBPs = zeros(no_dis)
  nxGBP, nyGBP = 2, 1

  for i in 1:no_dis
    ψ = randomITensorNetwork(s; link_space=chi)
    ψψ = norm_sqr_network(ψ; flatten=true, map_bra_linkinds=prime)
    combiners = linkinds_combiners(ψψ)
    ψψ = combine_linkinds(ψψ, combiners)
    mts = construct_initial_mts(ψψ, 1; init=(I...) -> allequal(I) ? 1 : 0)
    mts = update_all_mts(ψψ, mts, niters)

    mtsGBP = construct_initial_mts(ψψ, nxGBP * nyGBP; init=(I...) -> allequal(I) ? 1 : 0)
    mtsGBP = update_all_mts(ψψ, mtsGBP, niters)

    no_edges = length(edges(ψ))
    for e in edges(ψ)
      v1, v2 = src(e), dst(e)
      println("Working on Edge Between Site " * string(v1) * " and " * string(v2))
      #Probably shouldnt deepcopy for orthognalize to work here?
      # ψ_ortho = deepcopy(ψ)
      # for v in vertices(ψ_ortho)
      #   ψ_ortho = orthogonalize(deepcopy(ψ), v)
      # end
      ψ_ortho = orthogonalize(deepcopy(ψ), v1)
      no_ortho_rdm = approx_two_site_rdm(ψ, v1, v2, s)
      ortho_rdm = approx_two_site_rdm(ψ_ortho, v1, v2, s)
      actual_rdm = exact_two_site_rdm(ψ, v1, v2, s)
      trace_dist_SUs[i] += (1 / no_edges) * trace_dist(no_ortho_rdm, actual_rdm)
      trace_dist_ortho_SUs[i] += (1 / no_edges) * trace_dist(ortho_rdm, actual_rdm)
      SBP_rdm = two_site_rdm_bp(ψψ, ψ, mts, v1, v2, s, combiners)
      GBP_rdm = two_site_rdm_bp(ψψ, ψ, mtsGBP, v1, v2, s, combiners)
      trace_dist_SBPs[i] += (1 / no_edges) * trace_dist(SBP_rdm, actual_rdm)
      trace_dist_GBPs[i] += (1 / no_edges) * trace_dist(GBP_rdm, actual_rdm)
    end
    println(
      "Average Trace Distance between actual rdm and approx rdm using simple update is " *
      string(trace_dist_SUs[i]),
    )
    println(
      "Average Trace Distance between actual rdm and approx rdm using simple update with orthogonalisation is " *
      string(trace_dist_ortho_SUs[i]),
    )
    println(
      "Average Trace Distance between actual rdm and approx rdm using Simple Belief Propagation is " *
      string(trace_dist_SBPs[i]),
    )
    println(
      "Average Trace Distance between actual rdm and approx rdm using General Belief Propagation is " *
      string(trace_dist_GBPs[i]),
    )
  end

  return trace_dist_SUs, trace_dist_ortho_SUs, trace_dist_SBPs, trace_dist_GBPs
end

lx, ly = 4, 4
ITensors.disable_warn_order()
trace_dist_SUs, trace_dist_ortho_SUs, trace_dist_SBPs, trace_dist_GBPs = benchmark(lx, ly)
npzwrite(
  "../../../Documents/Data/ITensorNetworks/QuantumCircuits/RDMFidelitylx" *
  string(lx) *
  "ly" *
  string(ly) *
  ".npz";
  trace_dist_SUs=trace_dist_SUs,
  trace_dist_SBPs=trace_dist_SBPs,
  trace_dist_ortho_SUs=trace_dist_ortho_SUs,
  trace_dist_GBPs=trace_dist_GBPs,
)
