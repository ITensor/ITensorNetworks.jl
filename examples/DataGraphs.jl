using ITensors
using ITensorNetworks

#
# Helper functions
#
function vertextag(v::Tuple)
  return "$(v[1])×$(v[2])"
end

function edgetag(e)
  return "$(vertextag(src(e)))↔$(vertextag(dst(e)))"
end

χ, d = 5, 2
g = set_vertices(grid((2, 2)), (2, 2))

const Network{T} = DataGraph{T,T}

# Network of indices
is = Network{Vector{Index}}(g)
for e in edges(is)
  is[e] = [Index(χ, edgetag(e))]
end
for v in vertices(is)
  is[v] = [Index(d, vertextag(v))]
end

tn = Network{ITensor}(g)
for v in vertices(tn)
  siteinds = is[v]
  linkinds = [is[v => nv] for nv in neighbors(is, v)]
  tn[v] = ITensor(siteinds, linkinds...)
end

