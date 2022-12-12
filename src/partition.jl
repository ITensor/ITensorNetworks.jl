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

function partition(g::Graph, npartitions::Integer; backend=current_partitioning_backend(), kwargs...)
  return partition(Backend(backend), g, npartitions; kwargs...)
end

function partition(g::NamedGraph, npartitions::Integer; kwargs...)
  partitions = partition(parent_graph(g), npartitions; kwargs...)
  #[inv(vertex_to_parent_vertex(g))[v] for v in partitions]
  # TODO: output the reverse of this dictionary (a Vector of Vector
  # of the vertices in each partition).
  return Dictionary(vertices(g), partitions)
end

function partition(g::AbstractDataGraph, npartitions::Integer; kwargs...)
  return partition(underlying_graph(g), npartitions; kwargs...)
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
