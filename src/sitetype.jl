function siteind(sitetype::String, v::Tuple; kwargs...)
  return addtags(siteind(sitetype; kwargs...), ITensorNetworks.vertex_tag(v))
end

# naming collision of ITensors.addtags and addtags keyword in siteind system
function siteind(d::Integer, v::Tuple; addtags="", kwargs...)
  return ITensors.addtags(
    Index(d; tags="Site, $addtags", kwargs...), ITensorNetworks.vertex_tag(v)
  )
end

function siteinds(sitetypes::AbstractDictionary, g::AbstractGraph; kwargs...)
  is = IndsNetwork(g)
  for v in vertices(g)
    is[v] = [siteind(sitetypes[v], vertex_tag(v); kwargs...)]
  end
  return is
end

function siteinds(sitetype, g::AbstractGraph; kwargs...)
  return siteinds(Dictionary(vertices(g), fill(sitetype, nv(g))), g; kwargs...)
end

function siteinds(f::Function, g::AbstractGraph; kwargs...)
  return siteinds(Dictionary(vertices(g), map(v -> f(v), vertices(g))), g; kwargs...)
end
