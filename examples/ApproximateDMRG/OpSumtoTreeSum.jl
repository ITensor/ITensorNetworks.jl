using Compat
using ITensors
using Metis
using ITensorNetworks
using Random
using NamedGraphs
using NamedGraphs: rem_edge!
using Graphs
using KrylovKit
using LinearAlgebra
using Arpack
using Dictionaries
using SplitApplyCombine

using ITensorNetworks:
  rename_vertices_itn,
  compute_message_tensors,
  nested_graph_leaf_vertices,
  get_environment,
  contract_inner,
  calculate_contraction

maybe_only(x) = x
maybe_only(x::Tuple{T}) where {T} = only(x)

function squeeze(A::AbstractArray)
  keepdims = Tuple(i for i in size(A) if i != 1)
  return reshape(A, keepdims)
end

function TFI(g::DataGraph, h::Float64)
  os = OpSum()
  for e in edges(g)
    if (g[e] == 1)
      os += 1.0, "Sx", src(e), "Sx", dst(e)
    else
      os += 0.0, "Id", src(e), "Id", dst(e)
    end
  end

  for v in vertices(g)
    os += h, "Sz", v
  end

  return os
end

function convert_AbstractGraph_to_NamedGraph(g::AbstractGraph)
  g_out = NamedGraph(vertices(g))
  for e in edges(g)
    add_edge!(g_out, e)
  end
  return g_out
end

function union_trees_track_identity_edges(gs::Vector{NamedGraph})
  g_out = DataGraph(gs[1])
  for g in gs[2:length(gs)]
    g_out = union(g_out, DataGraph(g))
    e = vertices(g)[1] => vertices(g_out)[1]
    add_edge!(g_out, e)
    g_out[e] = 0
  end

  for e in edges(g_out)
    if (e ∉ keys(edge_data(g_out)))
      g_out[e] = 1
    end
  end

  return g_out
end

#Given a graph, split it into its connected connected_components, travserse them with an MST and construct the union
function reduce_graph(g::AbstractGraph)
  subsets = connected_components(g)
  gs = NamedGraph[]
  for subset in subsets
    g_red = g[subset]
    mst = bfs_tree(g_red, vertices(g_red)[1])
    push!(gs, convert_AbstractGraph_to_graph(mst))
  end

  return g_full = union_trees_track_identity_edges(gs)
end

function build_spanning_tree_cover(g::AbstractGraph)
  edges_collected = edgetype(g)[]
  gs_collected = DataGraph[]
  while (Set(edges_collected) != Set(edges(g)))
    g_red = remove_edges(g, edges_collected)
    g_red_MST = reduce_graph(g_red)
    edges_caught = edges(g_red_MST)[findall(
      ==(1), [g_red_MST[e] for e in edges(g_red_MST)]
    )]
    push!(edges_collected, edges_caught...)
    push!(gs_collected, g_red_MST)
  end

  return gs_collected
end

#NEED TO WORK ON THIS... SEEMS WEIRD (SUGGESTS MAP ISN'T POSDEF HERMITIAN BUT IT SHOULD BE...)
function update_vertex(
  ψ::ITensorNetwork, ttnos::Vector{TreeTensorNetwork}, v, nvertices_per_partition
)
  ψ = copy(ψ)
  ψ_cur = ψ[v]
  ψψ = dag(prime(ψ; sites=[])) ⊗ ψ
  ψψ_mts = compute_message_tensors(
    ψψ;
    vertex_groups=nested_graph_leaf_vertices(
      partition(
        partition(ψψ, group(v -> v[1], vertices(ψψ)));
        nvertices_per_partition=nvertices_per_partition,
      ),
    ),
  )
  ψenvs = get_environment(ψψ, ψψ_mts, [(v, 1), (v, 2)])

  Henvs = []

  for ttno in ttnos
    ψHiψ = rename_vertices_itn(dag(prime(ψ)) ⊗ ttno ⊗ ψ, v -> v[2] == 2 ? (v[1], 3) : v[1])
    ψHiψ_mts = compute_message_tensors(
      ψHiψ;
      vertex_groups=nested_graph_leaf_vertices(
        partition(
          partition(ψHiψ, group(v -> v[1], vertices(ψHiψ)));
          nvertices_per_partition=nvertices_per_partition,
        ),
      ),
    )
    push!(Henvs, get_environment(ψHiψ, ψHiψ_mts, [(v, 1), (v, 3)]))
  end

  c1 = cost_function(ψ_cur, Henvs, ψenvs)

  """THIS BLOCK IS FINE EXCEPT IT KEEPS RAISING POSDEF ERRORS?!?!"""
  # DMRG_Map_partial = partial(DRMG_map, Henvs, ψenvs)

  # try
  #     vals, vecs, info = KrylovKit.geneigsolve(DMRG_Map_partial,ψ_cur, 1, :SR; ishermitian=true, isposdef = true, maxiter = 100)
  #     ψ_cur  = vecs[1]
  # catch
  #     println("Had to Use a Random Vector")
  #     vals, vecs, info = KrylovKit.geneigsolve(DMRG_Map_partial,randomITensor(inds(ψ_cur)), 1, :SR; ishermitian=true, isposdef = true, maxiter = 100)
  #     ψ_cur  = vecs[1]
  # end

  #Lets do this matrix vector wise for now...

  sind = inds(ψ[v]; tags="Site")
  Id = ITensors.op("I", sind...)
  ψenvs_contracted = ITensors.contract(vcat(ψenvs, [Id]))
  row_inds, col_inds = inds(ψenvs_contracted; plev=0), inds(ψenvs_contracted; plev=1)
  C_row, C_col = combiner(row_inds), combiner(col_inds)
  ψenvs_contracted = (ψenvs_contracted * C_row) * C_col
  B = Matrix(ψenvs_contracted, inds(ψenvs_contracted))

  matrices = []
  for Henv in Henvs
    T = ITensors.contract(Henv; sequence=ITensors.optimal_contraction_sequence(Henv))
    T_contracted = (T * C_row) * C_col
    push!(matrices, Matrix(T_contracted, inds(ψenvs_contracted)))
  end
  A = sum(matrices)

  vcontracted = ψ[v] * C_row
  v_init = Array(vcontracted, inds(vcontracted))
  v_init = convert(Vector{ComplexF64}, v_init)
  A = convert(Array{ComplexF64}, A)
  B = convert(Array{ComplexF64}, B)

  S = inv(B + 1e-5 * I) * A
  eigvals, eigvecs = Arpack.eigs(S; nev=1, which=:SR, ritzvec=true, v0=v_init)
  eigvecs = squeeze(eigvecs)

  # #Problem solution is good... something really weird going on in translation below...
  ψ_rewrapped = ITensor(eigvecs, combinedind(C_row)) * C_row
  ψ_cur = copy(ψ_rewrapped)

  c2 = cost_function(ψ_cur, Henvs, ψenvs)

  return ψ_cur
end

function DMRG_sweep(
  ψ::ITensorNetwork,
  ttnos::Vector{TreeTensorNetwork},
  s::IndsNetwork,
  h::Float64;
  nvertices_per_partition=1,
  nsweeps=5,
  exact_energy_calc=true,
)
  ψ = copy(ψ)
  for i in 1:nsweeps
    if (!exact_energy_calc)
      E = TFI_energy_BP(ψ, s, h; nvertices_per_partition=nvertices_per_partition)
    else
      E = calc_energy(ψ, ttnos)
    end
    println("On Sweep " * string(i) * " energy is " * string(E))
    for v in vertices(ψ)
      ψ[v] = update_vertex(ψ, ttnos, v, nvertices_per_partition)
    end

    for v in reverse(vertices(ψ))
      ψ[v] = update_vertex(ψ, ttnos, v, nvertices_per_partition)
    end
  end

  if (!exact_energy_calc)
    E = TFI_energy_BP(ψ, s, h; nvertices_per_partition=nvertices_per_partition)
  else
    E = calc_energy(ψ, ttnos)
  end

  println("Finished with an Energy of " * string(E))

  return ψ
end

function calc_energy(ψ::ITensorNetwork, ttnos::Vector{TreeTensorNetwork})
  E = 0.0
  for ttno in ttnos
    ψHiψ = rename_vertices_itn(dag(prime(ψ)) ⊗ ttno ⊗ ψ, v -> v[2] == 2 ? (v[1], 3) : v[1])
    E += ITensors.contract(ψHiψ)[]
  end

  return E / contract_inner(ψ, ψ)
end

function TFI_energy_BP(
  ψ::ITensorNetwork,
  s::IndsNetwork,
  h::Float64;
  nvertices_per_partition=nvertices_per_partition,
)
  ψψ = ψ ⊗ dag(prime(ψ; sites=[]))
  ψψ_mts = compute_message_tensors(
    ψψ;
    vertex_groups=nested_graph_leaf_vertices(
      partition(
        partition(ψψ, group(v -> v[1], vertices(ψψ)));
        nvertices_per_partition=nvertices_per_partition,
      ),
    ),
  )
  gates = TFI_gates(s, h)
  e = 0
  for gate in gates
    qubits_to_act_on = vertices(s)[findall(
      i -> (length(commoninds(s[i], inds(gate))) != 0), vertices(s)
    )]
    num = calculate_contraction(
      ψψ,
      ψψ_mts,
      [(q, 1) for q in qubits_to_act_on];
      verts_tensors=[noprime!(prod([ψ[q] for q in qubits_to_act_on]) * gate)],
    )[]
    denom = calculate_contraction(ψψ, ψψ_mts, [(q, 1) for q in qubits_to_act_on])[]
    e += num / denom
  end
  return e
end

function TFI_gates(s::IndsNetwork, h)
  gates = ITensor[]
  for e in edges(s)
    hj = 1.0 * op("Sx", s[maybe_only(src(e))]) * op("Sx", s[maybe_only(dst(e))])
    push!(gates, hj)
  end

  for v in vertices(s)
    hj = h * op("Sz", s[v])
    push!(gates, hj)
  end

  return gates
end

function map_siteindsnetwork(s::IndsNetwork, smap::IndsNetwork)
  s = copy(s)
  for v in vertices(s)
    s[v] = smap[v]
  end

  return s
end

function DRMG_map(Henvs, ψvenvs::Vector{ITensor}, ψv::ITensor)
  return A_ψ(Henvs, ψv), B_ψ(ψvenvs, ψv)
end

function A_ψ(Henvs, ψv::ITensor)
  ψs = ITensor[]

  for Henv in Henvs
    H_tensors = vcat([ψv], Henv)
    push!(
      ψs,
      noprime(
        ITensors.contract(
          H_tensors; sequence=ITensors.optimal_contraction_sequence(H_tensors)
        ),
      ),
    )
  end

  return sum(ψs)
end

function B_ψ(ψvenvs::Vector{ITensor}, ψv::ITensor)
  tensors = vcat([ψv], ψvenvs)

  return noprime(
    ITensors.contract(tensors; sequence=ITensors.optimal_contraction_sequence(tensors))
  )
end

function cost_function(ψv::ITensor, Henvs, ψvenvs::Vector{ITensor})
  norm_tensors = vcat([ψv, noprime!(prime(dag(ψv)); tags="Site")], ψvenvs)
  norm = ITensors.contract(
    norm_tensors; sequence=ITensors.optimal_contraction_sequence(norm_tensors)
  )[]

  Hscalars = []

  count = 1
  for Henv in Henvs
    Hscalar_tensors = vcat([ψv, prime(dag(ψv))], Henv)
    push!(
      Hscalars,
      ITensors.contract(
        Hscalar_tensors; sequence=ITensors.optimal_contraction_sequence(Hscalar_tensors)
      )[],
    )
    count += 1
  end

  return sum(Hscalars) / norm
end

function TFI_DMRG_backend(dims, chi_max, h, adj_mat, name_map, s::IndsNetwork)
  L = prod(dims)
  sites = siteinds("S=1/2", L)
  init_state = [isodd(i) ? "Up" : "Dn" for i in 1:L]

  sweeps = Sweeps(10)
  setmaxdim!(sweeps, trunc(Int, chi_max))
  setcutoff!(sweeps, 1E-10)

  ampo = OpSum()
  for i in 1:L
    ampo += h, "Sz", i
    for j in (i + 1):L
      ampo += 1 * adj_mat[i, j], "Sx", i, "Sx", j
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

function build_ttnos_TFI(g::NamedGraph, h::Float64, s::IndsNetwork)
  data_graphs = build_spanning_tree_cover(g)
  ttnos = TreeTensorNetwork[]
  for dg in data_graphs
    mapped_s = map_siteindsnetwork(siteinds("S=1/2", convert_AbstractGraph_to_graph(dg)), s)
    ops = TFI(dg, h / length(data_graphs))
    push!(ttnos, TTN(ops, mapped_s))
  end

  return ttnos
end

partial = (f, a...; c...) -> (b...) -> f(a..., b...; c...)

n = 2
dims = (n, n, n)
g = named_grid(dims)
g = convert_AbstractGraph_to_graph(bfs_tree(g, vertices(g)[1]))
s = siteinds("S=1/2", g)
chi = 250
h = 1.2

adj_mat, name_map = create_adj_mat(g)
ψ_DMRG = TFI_DMRG_backend(dims, chi, h, adj_mat, name_map, s)

ttnos = build_ttnos_TFI(g, h, s)

println("Now lets sweep using our new DMRG routine, with BP constructed environments")
ITensors.disable_warn_order()
ψ_init = randomITensorNetwork(s; link_space=2)
ψGS = DMRG_sweep(ψ_init, ttnos, s, h; nvertices_per_partition=2, exact_energy_calc=false)
