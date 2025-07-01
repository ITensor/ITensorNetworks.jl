import Graphs: dst, edges, src, vertices
import NamedGraphs as ng

function compute_adjacencies(G)
  adj = Dict(v => Vector{ng.vertextype(G)}() for v in vertices(G))
  for e in edges(G)
    push!(adj[src(e)], dst(e))
    push!(adj[dst(e)], src(e))
  end
  return adj
end

function euler_tour_edges(G, start_vertex)
  adj = compute_adjacencies(G)
  etype = ng.edgetype(G)
  vtype = ng.vertextype(G)
  visited = Set{Tuple{vtype,vtype}}()
  tour = Vector{etype}()
  stack = [start_vertex]
  while !isempty(stack)
    u = stack[end]
    pushed = false
    for v in adj[u]
      if (u, v) ∉ visited
        push!(visited, (u, v))
        push!(visited, (v, u))
        push!(tour, etype(u => v))
        push!(stack, v)
        pushed = true
        break  # handle one neighbor at a time
      end
    end
    if !pushed
      pop!(stack)
      if !isempty(stack)
        v = stack[end]
        push!(tour, etype(u => v))  # Backtracking step
      end
    end
  end
  return tour
end

function euler_tour_vertices(G, start_vertex)
  edges = euler_tour_edges(G, start_vertex)
  isempty(edges) && return Vector{eltype(vertices(G))}[]
  return [src(edges[1]), dst.(edges)...]
end
