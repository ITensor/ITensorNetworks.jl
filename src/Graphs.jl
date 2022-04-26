function incident_edges(graph::AbstractGraph, vertex...)
  return [edgetype(graph)(to_vertex(graph, vertex...), neighbor_vertex) for neighbor_vertex in neighbors(graph, vertex...)]
end
