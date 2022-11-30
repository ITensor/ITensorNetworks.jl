#Given a graph g, form 'nsubgraphs' subgraphs. Try to keep all subgraphs the same size and minimise edges cut between them
#Returns a datagraph where each vertex contains the list of vertices involved in that subgraph. The edges state which subgraphs are connected
#KaHyPar needs to be installed for this function to work
function formsubgraphs(g::NamedDimGraph, nsubgraphs::Int64)
  ps = partition(g, nsubgraphs; configuration=:edge_cut, imbalance=0.0)

  dg_subgraphs = DataGraph{Vector{Tuple},Any}(NamedDimGraph([(i,) for i in 1:nsubgraphs]))
  for s in 1:nsubgraphs
    dg_subgraphs[(s,)] = [v for v in vertices(g) if ps[v] == s]
  end

  for e in edges(g)
    v1, v2 = src(e), dst(e)
    s1, s2 = find_subgraph(v1, dg_subgraphs), find_subgraph(v2, dg_subgraphs)
    if (!has_edge(dg_subgraphs, s1, s2) && s1 != s2)
      add_edge!(dg_subgraphs, s1, s2)
    end
  end

  return dg_subgraphs
end

#Given a graph g on a d-dimensional grid of size Ls[1] x Ls[2] x ..., form subgraphs of size ls[1] x ls[2] in a regular fashion
#Return the subgraphs (1... npartitions) with their contained vertices. Also return a dictionary of the subgraphs connected to each sgraph
function formsubgraphs_grid(g::NamedDimGraph, Ls::Vector{Int64}, ls::Vector{Int64})
  lengths = Ls ./ ls
  ps = Dict{Tuple,Int64}()
  for v in vertices(g)
    pos = []
    count = 1
    for i in 1:length(v)
      push!(pos, ceil(Int64, v[i] / ls[i]))
      if (pos[i] == 1)
        p = 1
      else
        p = prod(lengths[1:(i - 1)])
      end
      count = Int(count + (pos[i] - 1) * p)
    end
    ps[v] = count
  end

  nsubgraphs = Int(prod(lengths))

  dg_subgraphs = DataGraph{Vector{Tuple},Any}(NamedDimGraph([(i,) for i in 1:nsubgraphs]))
  for s in 1:nsubgraphs
    dg_subgraphs[(s,)] = [v for v in vertices(g) if ps[v] == s]
  end

  for e in edges(g)
    v1, v2 = src(e), dst(e)
    s1, s2 = find_subgraph(v1, dg_subgraphs), find_subgraph(v2, dg_subgraphs)
    if (!has_edge(dg_subgraphs, s1, s2) && s1 != s2)
      add_edge!(dg_subgraphs, s1, s2)
    end
  end

  return dg_subgraphs
end

#Find the subgraph in subgraph data_graph which contains vertex
function find_subgraph(vertex::Tuple, dg_subgraphs::DataGraph)
  for v in vertices(dg_subgraphs)
    verts = dg_subgraphs[v]
    if vertex in verts
      return v
    end
  end
end
