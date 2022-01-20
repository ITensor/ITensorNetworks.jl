function siteind(sitetype::String, v::Tuple; kwargs...)
  return addtags(siteind(sitetype; kwargs...), ITensorNetworks.vertex_tag(v))
end

function siteinds(sitetype::AbstractString, g::AbstractGraph; kwargs...)
  is = IndsNetwork(g)
  for v in vertices(is)
    is[v] = [siteind(sitetype, v; kwargs...)]
  end
  return is
end
