#=
    Metis.partition(G, n; alg = :KWAY)

Partition the graph `G` in `n` parts.
The partition algorithm is defined by the `alg` keyword:
 - :KWAY: multilevel k-way partitioning
 - :RECURSIVE: multilevel recursive bisection
=#
function partition(g::Metis.Graph, npartitions::Integer)
  return Metis.partition(g, npartitions; alg=:KWAY)
end

function partition(g::Graph, npartitions::Integer)
  return partition(Metis.graph(adjacency_matrix(g)), npartitions)
end

function partition(g::CustomVertexGraph, npartitions::Integer)
  partitions = partition(parent_graph(g), npartitions)
  #[inv(vertex_to_parent_vertex(g))[v] for v in partitions]
  # TODO: output the reverse of this dictionary (a Vector of Vector
  # of the vertices in each partition).
  return Dictionary(vertices(g), partitions)
end

function partition(g::AbstractDataGraph, npartitions::Integer)
  return partition(underlying_graph(g), npartitions)
end

