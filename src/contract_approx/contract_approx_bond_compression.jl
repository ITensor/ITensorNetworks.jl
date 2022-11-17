"""
$(TYPEDSIGNATURES)

Approximately contract a tensor network using the algorithm `alg`.
"""
function contract_approx(tn::AbstractITensorNetwork; alg="bond_compression", kwargs...)
  return contract_approx(Algorithm(alg), tn; kwargs...)
end

"""
    vertex_or_vertex_set(x)

Returns a vertex or a set of vertices.
"""
function vertex_or_vertex_set(x::Vertex)
  return x
end
function vertex_or_vertex_set(x::AbstractVector)
  return Set(Leaves(x))
end

unvertex(x::Vertex) = x.vertex
unvertex(x::Set) = Set(unvertex.(x))

"""
Bond compression algorithm from https://arxiv.org/abs/2206.07044.

For grid networks, a good sequence is:
```julia
source_vertex = argmax(closeness_centrality(tn))
spanning_tree = bfs_tree(tn, source_vertex)
contraction_vertices = post_order_dfs_vertices(spanning_tree, source_vertex)
sequence = reduce((v1, v2) -> [v1, v2], contraction_vertices)
```
"""
function contract_approx(::Algorithm"bond_compression", tn::AbstractITensorNetwork; sequence, maxdim, cutoff)
  # Recurse through the contraction sequence
  # Make an edge list from a contraction sequence!
  print_tree(deepmap(Vertex, sequence); maxdepth=10)
  cache = Dictionary()
  contractions = collect(PostOrderDFS(deepmap(Vertex, sequence)))
  for contraction in contractions
    @show contraction
    if contraction isa AbstractVector
      @assert length(contraction) == 2
      v1 = vertex_or_vertex_set(contraction[1])
      v2 = vertex_or_vertex_set(contraction[2])
      new_v = Set(v1) ∪ Set(v2)
      @show tn
      tn = contract(tn, unvertex(v1) => unvertex(v2); new_vertex=unvertex(new_v))
      @show tn
      # res = cache[contraction[1]] * cache[contraction[2]]
      # set!(cache, contraction, res)
    elseif contraction isa Vertex
      # Do nothing, we are at a leaf of the contraction
      # sequence tree.
    else
      error("Invalid node type $(typeof(contraction)))")
    end
  end
  return tn
end

# Reference implementation
function _contract_approx_reference(::Algorithm"bond_compression", tn::AbstractITensorNetwork; sequence, maxdim, cutoff)
  # Recurse through the contraction sequence
  # Make an edge list from a contraction sequence!
  print_tree(deepmap(Vertex, sequence); maxdepth=10)
  cache = Dictionary()
  contractions = collect(PostOrderDFS(deepmap(Vertex, sequence)))
  for contraction in contractions
    @show contraction
    if contraction isa Vertex
      set!(cache, contraction, tn[contraction.vertex])
    elseif contraction isa AbstractVector
      @assert length(contraction) == 2
      res = cache[contraction[1]] * cache[contraction[2]]
      set!(cache, contraction, res)
    else
      error("Invalid node type $(typeof(contraction)))")
    end
  end
  return cache[last(contractions)]
end
