struct Backend{T} end
Backend(s) = Backend{Symbol(s)}()
macro Backend_str(s)
  return :(Backend{$(Expr(:quote, Symbol(s)))})
end

# https://github.com/kahypar/KaHyPar.jl/issues/20
KaHyPar.HyperGraph(g::Graph) = incidence_matrix(g)

# configurations = readdir(joinpath(pkgdir(KaHyPar), "src", "config"))
#  "cut_kKaHyPar_sea20.ini"
#  "cut_rKaHyPar_sea20.ini"
#  "km1_kKaHyPar-E_sea20.ini"
#  "km1_kKaHyPar_eco_sea20.ini"
#  "km1_kKaHyPar_sea20.ini"
#  "km1_rKaHyPar_sea20.ini"
#
kahypar_configurations = Dict([
  (objective="edge_cut", alg="kway") => "cut_kKaHyPar_sea20.ini",
  (objective="edge_cut", alg="recursive") => "cut_rKaHyPar_sea20.ini",
  (objective="connectivity", alg="kway") => "km1_kKaHyPar_sea20.ini",
  (objective="connectivity", alg="recursive") => "km1_rKaHyPar_sea20.ini",
])

# default_configuration => "cut_kKaHyPar_sea20.ini"
# :edge_cut => "cut_kKaHyPar_sea20.ini"
# :connectivity => "km1_kKaHyPar_sea20.ini"
# imbalance::Number=0.03
function partition(
  ::Backend"KaHyPar",
  g::Graph,
  npartitions::Integer;
  objective="edge_cut",
  alg="kway",
  configuration=nothing,
  kwargs...,
)
  if isnothing(configuration)
    configuration = joinpath(
      pkgdir(KaHyPar),
      "src",
      "config",
      kahypar_configurations[(objective=objective, alg=alg)],
    )
  end
  partitions = @suppress KaHyPar.partition(g, npartitions; configuration, kwargs...)
  return partitions .+ 1
end

metis_algs = Dict(["kway" => :KWAY, "recursive" => :RECURSIVE])

function partition(::Backend"Metis", g::Graph, npartitions::Integer; alg="kway", kwargs...)
  metis_alg = metis_algs[alg]
  partitions = Metis.partition(g, npartitions::Integer; alg=metis_alg, kwargs...)
  return Int.(partitions)
end

function partition(g::Graph, npartitions::Integer; backend="KaHyPar", kwargs...)
  return partition(Backend(backend), g, npartitions; kwargs...)
end

function partition(g::NamedDimGraph, npartitions::Integer; kwargs...)
  partitions = partition(parent_graph(g), npartitions; kwargs...)
  #[inv(vertex_to_parent_vertex(g))[v] for v in partitions]
  # TODO: output the reverse of this dictionary (a Vector of Vector
  # of the vertices in each partition).
  return Dictionary(vertices(g), partitions)
end

function partition(g::AbstractDataGraph, npartitions::Integer; kwargs...)
  return partition(underlying_graph(g), npartitions; kwargs...)
end

## #=
##     Metis.partition(G, n; alg = :KWAY)
## 
## Partition the graph `G` in `n` parts.
## The partition algorithm is defined by the `alg` keyword:
##  - :KWAY: multilevel k-way partitioning
##  - :RECURSIVE: multilevel recursive bisection
## =#
## function partition(g::Metis.Graph, npartitions::Integer)
##   return Metis.partition(g, npartitions; alg=:KWAY)
## end
## 
## function partition(g::Graph, npartitions::Integer)
##   return partition(Metis.graph(adjacency_matrix(g)), npartitions)
## end
