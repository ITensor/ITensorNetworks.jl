set_partitioning_backend!(Backend"Metis"())

"""
    partition_vertices(::Backend"Metis", g::AbstractGraph, npartitions::Integer; alg="recursive")

Partition the graph `G` in `n` parts.
The partition algorithm is defined by the `alg` keyword:
 - :KWAY: multilevel k-way partitioning
 - :RECURSIVE: multilevel recursive bisection
"""
function partition_vertices(::Backend"Metis", g::SimpleGraph, npartitions::Integer; alg="recursive", kwargs...)
  metis_alg = metis_algs[alg]
  partitions = Metis.partition(g, npartitions; alg=metis_alg, kwargs...)
  return groupfind(Int.(partitions))
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
