"""
Graph partitioning backend
"""
struct Backend{T} end

Backend(s::Symbol) = Backend{s}()
Backend(s::String) = Backend(Symbol(s))
Backend(backend::Backend) = backend

macro Backend_str(s)
  return :(Backend{$(Expr(:quote, Symbol(s)))})
end

"""
Current default graph partitioning backend
"""
const CURRENT_PARTITIONING_BACKEND = Ref{Union{Missing,Backend}}(missing)

"""
Get the graph partitioning backend
"""
current_partitioning_backend() = CURRENT_PARTITIONING_BACKEND[]

"""
Set the graph partitioning backend
"""
function set_partitioning_backend!(backend::Union{Missing,Backend,String})
  CURRENT_PARTITIONING_BACKEND[] = Backend(backend)
  return nothing
end

# KaHyPar configuration options
#
# configurations = readdir(joinpath(pkgdir(KaHyPar), "src", "config"))
#  "cut_kKaHyPar_sea20.ini"
#  "cut_rKaHyPar_sea20.ini"
#  "km1_kKaHyPar-E_sea20.ini"
#  "km1_kKaHyPar_eco_sea20.ini"
#  "km1_kKaHyPar_sea20.ini"
#  "km1_rKaHyPar_sea20.ini"
#
const kahypar_configurations = Dict([
  (objective="edge_cut", alg="kway") => "cut_kKaHyPar_sea20.ini",
  (objective="edge_cut", alg="recursive") => "cut_rKaHyPar_sea20.ini",
  (objective="connectivity", alg="kway") => "km1_kKaHyPar_sea20.ini",
  (objective="connectivity", alg="recursive") => "km1_rKaHyPar_sea20.ini",
])

# Metis configuration options
const metis_algs = Dict(["kway" => :KWAY, "recursive" => :RECURSIVE])

function _npartitions(
  g::AbstractGraph, npartitions::Integer, nvertices_per_partition::Nothing
)
  return npartitions
end

function _npartitions(
  g::AbstractGraph, npartitions::Nothing, nvertices_per_partition::Integer
)
  return nv(g) ÷ nvertices_per_partition
end

function _npartitions(g::AbstractGraph, npartitions::Int, nvertices_per_partition::Int)
  return error("Can't specify both `npartitions` and `nvertices_per_partition`")
end

function _npartitions(
  g::AbstractGraph, npartitions::Nothing, nvertices_per_partition::Nothing
)
  return error("Must specify either `npartitions` or `nvertices_per_partition`")
end

function subgraph_vertices(
  g::Graph;
  npartitions=nothing,
  nvertices_per_partition=nothing,
  backend=current_partitioning_backend(),
  kwargs...,
)
  #Metis cannot handle the edge case npartitions = 1, so we will fix it here for now
  if (_npartitions(g, npartitions, nvertices_per_partition) == 1)
    return group(v -> 1, collect(vertices(g)))
  end

  return subgraph_vertices(
    Backend(backend), g, _npartitions(g, npartitions, nvertices_per_partition); kwargs...
  )
end

function subgraph_vertices(
  g::NamedGraph; npartitions=nothing, nvertices_per_partition=nothing, kwargs...
)
  vertex_partitions = subgraph_vertices(
    parent_graph(g); npartitions, nvertices_per_partition, kwargs...
  )
  #[inv(vertex_to_parent_vertex(g))[v] for v in partitions]
  # TODO: output the reverse of this dictionary (a Vector of Vector
  # of the vertices in each partition).
  # return Dictionary(vertices(g), partitions)
  return [
    parent_vertices_to_vertices(g, vertex_partition) for
    vertex_partition in vertex_partitions
  ]
end

function subgraph_vertices(
  g::AbstractDataGraph; npartitions=nothing, nvertices_per_partition=nothing, kwargs...
)
  return subgraph_vertices(
    underlying_graph(g); npartitions, nvertices_per_partition, kwargs...
  )
end

"""
    partition_vertices(g::AbstractGraph, subgraph_vertices::Vector)

Given a graph (`g`) and groups of vertices defining subgraphs of that
graph (`subgraph_vertices`), return a DataGraph storing the subgraph
vertices on the vertices of the graph and with edges denoting
which subgraphs of the original graph have edges connecting them, along with
edge data storing the original edges that were connecting the subgraphs.
"""
function partition_vertices(g::AbstractGraph, subgraph_vertices)
  partitioned_vertices = DataGraph(
    NamedGraph(eachindex(subgraph_vertices)), Dictionary(subgraph_vertices)
  )
  for e in edges(g)
    s1 = findfirst_on_vertices(
      subgraph_vertices -> src(e) ∈ subgraph_vertices, partitioned_vertices
    )
    s2 = findfirst_on_vertices(
      subgraph_vertices -> dst(e) ∈ subgraph_vertices, partitioned_vertices
    )
    if (!has_edge(partitioned_vertices, s1, s2) && s1 ≠ s2)
      add_edge!(partitioned_vertices, s1, s2)
      partitioned_vertices[s1 => s2] = Vector{edgetype(g)}()
    end
    if has_edge(partitioned_vertices, s1, s2)
      push!(partitioned_vertices[s1 => s2], e)
    end
  end
  return partitioned_vertices
end

"""
    partition_vertices(g::AbstractGraph; npartitions, nvertices_per_partition, kwargs...)

Given a graph `g`, partition the vertices of `g` into 'npartitions' partitions
or into partitions with `nvertices_per_partition` vertices per partition.
Try to keep all subgraphs the same size and minimise edges cut between them
Returns a datagraph where each vertex contains the list of vertices involved in that subgraph. The edges state which subgraphs are connected.
A graph partitioning backend such as Metis or KaHyPar needs to be installed for this function to work.
"""
function partition_vertices(
  g::AbstractGraph; npartitions=nothing, nvertices_per_partition=nothing, kwargs...
)
  return partition_vertices(
    g, subgraph_vertices(g; npartitions, nvertices_per_partition, kwargs...)
  )
end

"""
Find all vertices `v` such that `f(graph[v]) == true`
"""
function findall_on_vertices(f::Function, graph::AbstractDataGraph)
  return findall(f, vertex_data(graph))
end

"""
Find the vertex `v` such that `f(graph[v]) == true`
"""
function findfirst_on_vertices(f::Function, graph::AbstractDataGraph)
  return findfirst(f, vertex_data(graph))
end

"""
Find all edges `e` such that `f(graph[e]) == true`
"""
function findall_on_edges(f::Function, graph::AbstractDataGraph)
  return findall(f, edge_data(graph))
end

"""
Find the edge `e` such that `f(graph[e]) == true`
"""
function findfirst_on_edges(f::Function, graph::AbstractDataGraph)
  return findfirst(f, edge_data(graph))
end

"""
Find the subgraph which contains the specified vertex.

TODO: Rename something more general, like:

findfirst_in_vertex_data(item, graph::AbstractDataGraph)
"""
function find_subgraph(vertex, subgraphs::DataGraph)
  return findfirst_on_vertices(subgraph -> vertex ∈ vertices(subgraph), subgraphs)
end

function subgraphs(g::AbstractSimpleGraph, subgraph_vertices)
  return subgraphs(NamedGraph(g), subgraph_vertices)
end

"""
    subgraphs(g::AbstractGraph, subgraph_vertices)

Return a collection of subgraphs of `g` defined by the subgraph
vertices `subgraph_vertices`.
"""
function subgraphs(g::AbstractGraph, subgraph_vertices)
  return map(vs -> subgraph(g, vs), subgraph_vertices)
end

"""
    subgraphs(g::AbstractGraph; npartitions::Integer, kwargs...)

Given a graph `g`, partition `g` into `npartitions` partitions
or into partitions with `nvertices_per_partition` vertices per partition,
returning a list of subgraphs.
Try to keep all subgraphs the same size and minimise edges cut between them.
A graph partitioning backend such as Metis or KaHyPar needs to be installed for this function to work.
"""
function subgraphs(
  g::AbstractGraph; npartitions=nothing, nvertices_per_partition=nothing, kwargs...
)
  return subgraphs(g, subgraph_vertices(g; npartitions, nvertices_per_partition, kwargs...))
end

function partition(g::AbstractSimpleGraph, subgraph_vertices)
  return partition(NamedGraph(g), subgraph_vertices)
end

function partition(g::AbstractGraph, subgraph_vertices)
  partitioned_graph = DataGraph(
    NamedGraph(eachindex(subgraph_vertices)), subgraphs(g, Dictionary(subgraph_vertices))
  )
  for e in edges(g)
    s1 = findfirst_on_vertices(subgraph -> src(e) ∈ vertices(subgraph), partitioned_graph)
    s2 = findfirst_on_vertices(subgraph -> dst(e) ∈ vertices(subgraph), partitioned_graph)
    if (!has_edge(partitioned_graph, s1, s2) && s1 ≠ s2)
      add_edge!(partitioned_graph, s1, s2)
      partitioned_graph[s1 => s2] = Dictionary(
        [:edges, :edge_data],
        [Vector{edgetype(g)}(), Dictionary{edgetype(g),edge_data_type(g)}()],
      )
    end
    if has_edge(partitioned_graph, s1, s2)
      push!(partitioned_graph[s1 => s2][:edges], e)
      if isassigned(g, e)
        set!(partitioned_graph[s1 => s2][:edge_data], e, g[e])
      end
    end
  end
  return partitioned_graph
end

"""
    partition(g::AbstractGraph; npartitions::Integer, kwargs...)
    partition(g::AbstractGraph, subgraph_vertices)

Given a graph `g`, partition `g` into `npartitions` partitions
or into partitions with `nvertices_per_partition` vertices per partition.
The partitioning tries to keep all subgraphs the same size and minimize
edges cut between them.

Alternatively, specify a desired partitioning with a collection of sugraph
vertices.

Returns a data graph where each vertex contains the corresponding subgraph as vertex data.
The edges indicates which subgraphs are connected, and the edge data stores a dictionary
with two fields. The field `:edges` stores a list of the edges of the original graph
that were connecting the two subgraphs, and `:edge_data` stores a dictionary
mapping edges of the original graph to the data living on the edges of the original
graph, if it existed.

Therefore, one should be able to extract that data and recreate the original
graph from the results of `partition`.

A graph partitioning backend such as Metis or KaHyPar needs to be installed for this function to work
if the subgraph vertices aren't specified explicitly.
"""
function partition(
  g::AbstractGraph;
  npartitions=nothing,
  nvertices_per_partition=nothing,
  subgraph_vertices=nothing,
  kwargs...,
)
  if count(isnothing, (npartitions, nvertices_per_partition, subgraph_vertices)) != 2
    error(
      "Error: Cannot give multiple/ no partitioning options. Please specify exactly one."
    )
  end

  if isnothing(subgraph_vertices)
    subgraph_vertices = ITensorNetworks.subgraph_vertices(
      g; npartitions, nvertices_per_partition, kwargs...
    )
  end

  return partition(g, subgraph_vertices)
end

"""Given a nested graph fetch all the vertices down to the lowest levels and return the grouping
at the highest level. Keyword argument is used to state whether we are at the top"""
function nested_graph_leaf_vertices(g; toplevel=true)
  vertex_groups = []
  for v in vertices(g)
    if hasmethod(vertices, Tuple{typeof(g[v])})
      if !toplevel
        push!(vertex_groups, nested_graph_leaf_vertices(g[v]; toplevel=false)...)
      else
        push!(vertex_groups, nested_graph_leaf_vertices(g[v]; toplevel=false))
      end
    else
      push!(vertex_groups, [v]...)
    end
  end

  return vertex_groups
end
