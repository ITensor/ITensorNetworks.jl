function partition(::Backend"Metis", g::Graph, npartitions::Integer; alg="kway", kwargs...)
  metis_alg = metis_algs[alg]
  partitions = Metis.partition(g, npartitions::Integer; alg=metis_alg, kwargs...)
  return Int.(partitions)
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
