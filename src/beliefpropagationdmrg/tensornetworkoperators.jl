using NamedGraphs: NamedGraph, AbstractGraph, vertices, edges, NamedEdge
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.GraphsExtensions:
  forest_cover,
  connected_components,
  add_edges,
  add_edge,
  add_vertices,
  rem_edges,
  decorate_graph_edges,
  rename_vertices

using ITensors: OpSum, siteinds, sites, inds, combiner
using ITensors.NDTensors: array
using ITensorNetworks:
  AbstractITensorNetwork, ITensorNetwork, IndsNetwork, underlying_graph, ttn, maxlinkdim
using ITensorNetworks.ModelHamiltonians: ising
using Dictionaries

using LinearAlgebra: eigvals

function connect_forests(s::IndsNetwork)
  g = underlying_graph(s)
  fs = forest_cover(g)
  fs_connected = Tuple{IndsNetwork,Vector{NamedEdge}}[]
  for f in fs
    s_f = copy(s)
    verts = vertices(f)
    es = edges(f)
    c_f = NamedGraph[f[vs] for vs in connected_components(f)]
    dummy_es = NamedEdge[
      NamedEdge(first(vertices(c_f[i])) => first(vertices(c_f[i + 1]))) for
      i in 1:(length(c_f) - 1)
    ]
    s_f = rem_edges(s_f, edges(s_f))
    s_f = add_edges(s_f, vcat(es, dummy_es))
    push!(fs_connected, (s_f, dummy_es))
  end
  return fs_connected
end

function opsum_to_tno(
  s::IndsNetwork, H::OpSum; cutoff::Float64=1e-14, insert_dummy_inds=false
)
  s_fs_connected = connect_forests(s)
  no_forests = length(s_fs_connected)
  tnos = ITensorNetwork[]
  for (s_f, dummy_edges) in s_fs_connected
    real_es = setdiff(edges(s_f), dummy_edges)
    new_opsum = OpSum()
    for term in H
      if length(sites(term)) == 1 && !iszero(first(term.args))
        new_opsum += (1.0 / no_forests) * term
      elseif length(sites(term)) == 2 && !iszero(first(term.args))
        e = NamedEdge(first(sites(term)) => last(sites(term)))
        if (e ∈ real_es || reverse(e) ∈ real_es)
          new_opsum += term
        end
      end
    end
    tno = truncate(ttn(new_opsum, s_f); cutoff)
    tno = ITensorNetwork(tno)
    if insert_dummy_inds
      tno = insert_linkinds(tno, setdiff(edges(s), edges(tno)))
    end
    push!(tnos, tno)
  end

  return tnos
end
