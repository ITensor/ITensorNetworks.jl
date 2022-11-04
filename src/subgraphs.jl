#Given a graph g, form 'nsubgraphs' subgraphs. Try to keep all subgraphs the same size and minimise edges cut between them
#Return the subgraphs (1... nsubgraphs) with their contained vertices. Also return a dictionary of the subgraphs connected to each sgraph
#KaHyPar needs to be installed for this function
function formsubgraphs(g::NamedDimGraph, nsubgraphs::Int64)
  ps = partition(g, nsubgraphs; configuration=:edge_cut, imbalance=0.0)

  subgraphconns = Dict{Int,Vector{Int}}()
  subgraphs = Dict{Int,Vector{Tuple}}()
  for s in 1:nsubgraphs
    subgraphs[s] = [v for v in vertices(g) if ps[v] == s]
  end

  subgraphadjmat = zeros(Int, nsubgraphs, nsubgraphs)
  es = edges(g)

  for i in 1:nsubgraphs
    for j in (i + 1):nsubgraphs
      sgraph1, sgraph2 = subgraphs[i], subgraphs[j]
      for v1 in sgraph1
        for v2 in sgraph2
          if (find_edge(es, v1, v2) != 0)
            subgraphadjmat[i, j], subgraphadjmat[j, i] = 1, 1
          end
        end
      end
    end

    row = subgraphadjmat[i, :]
    inds = findall(!iszero, row)
    subgraphconns[i] = inds
  end

  return subgraphs, subgraphconns
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

  subgraphconns = Dict{Int,Vector{Int}}()
  subgraphs = Dict{Int,Vector{Tuple}}()
  for s in 1:nsubgraphs
    subgraphs[s] = [v for v in vertices(g) if ps[v] == s]
  end

  subgraphadjmat = zeros(Int, nsubgraphs, nsubgraphs)
  es = edges(g)

  for i in 1:nsubgraphs
    for j in (i + 1):nsubgraphs
      sgraph1, sgraph2 = subgraphs[i], subgraphs[j]
      for v1 in sgraph1
        for v2 in sgraph2
          if (find_edge(es, v1, v2) != 0)
            subgraphadjmat[i, j], subgraphadjmat[j, i] = 1, 1
          end
        end
      end
    end

    row = subgraphadjmat[i, :]
    inds = findall(!iszero, row)
    subgraphconns[i] = inds
  end

  return subgraphs, subgraphconns
end

#Find the subgraph in the list of subgraphs which contains the vertex v 
function find_subgraph(vertex::Tuple, subgraphs::Dict{Int,Vector{Tuple}})
  for i in 1:length(subgraphs)
    for v in subgraphs[i]
      if (v == vertex)
        return i
      end
    end
  end
end

#CHECK IF TWO VERTICES ARE INT HE SAME SUBGRAPH
function in_same_subgraph(subgraphs::Dict{Int,Vector{Tuple}}, v1::Tuple, v2::Tuple)
  if find_subgraph(v1, subgraphs) == find_subgraph(v2, subgraphs)
    return true
  else
    return false
  end
end
