using ITensors
using ITensorNetworks
using Graphs
using ITensorUnicodePlots

#
# Add-on definitions
#

function ITensors.siteind(site_type::String, v::Tuple; kwargs...)
  return addtags(siteind(site_type; kwargs...), ITensorNetworks.vertex_tag(v))
end

function ITensors.siteinds(site_type::AbstractString, g::AbstractGraph; kwargs...)
  is = IndsNetwork(g)
  for v in vertices(is)
    is[v] = [siteind(site_type, v; kwargs...)]
  end
  return is
end

dims = (3, 3)
g = set_vertices(grid(dims), dims)

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

χ = 5
ψ = ITensorNetwork(s; link_space=χ)

ψt = itensors(ψ)
@visualize ψt

nothing
