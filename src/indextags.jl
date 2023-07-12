# Helper functions
vertex_tag(v) = "$v"

function vertex_tag(v::Tuple)
  t = "$(vertex_tag(first(v)))"
  for vn in Base.tail(v)
    t *= "×$(vertex_tag(vn))"
  end
  return t
end

edge_tag(e::Pair) = edge_tag(NamedEdge(e))

function edge_tag(e)
  return "$(vertex_tag(src(e))),$(vertex_tag(dst(e)))"
end

function vertex_index(v, vertex_space)
  return Index(vertex_space; tags=vertex_tag(v))
end

function edge_index(e, edge_space)
  return Index(edge_space; tags=edge_tag(e))
end
