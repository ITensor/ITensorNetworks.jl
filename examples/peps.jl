using ITensors
using ITensorNetworks
using Graphs
using ITensorGLMakie

g = CustomVertexGraph(grid((3, 3)), (3, 3))

function heisenberg(g::AbstractGraph)
  # TODO: os = Sum{Op}()
  os = OpSum()
  for e in edges(g)
    os += 1/2, "S⁺", src(e), "S⁻", dst(e)
    os += 1/2, "S⁺", src(e), "S⁻", dst(e)
    os += "Sᶻ", src(e), "Sᶻ", dst(e)
  end
  return os
end

ℋ = heisenberg(g)
s = siteinds("S=1/2", g)
