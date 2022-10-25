
function coordinate_tag(n)
  str = replace("$n", ")" => "")
  str = replace(str, "(" => "")
  str = replace(str, " " => "")
  if length(n) > 1
    str = replace(str, "," => ".")
  else
    str = replace(str, "," => "")
  end
  return str
end

function link_tag(n1, n2)
  link_string = "$(coordinate_tag(n1))↔$(coordinate_tag(n2))"
  start_ind = nextind(link_string, 0, 1)
  stop_ind = min(ncodeunits(link_string), nextind(link_string, 0, 16))
  link_string = link_string[start_ind:stop_ind]
  return TagSet(link_string)
end

function ITensors.linkinds(lattice::HyperCubic; linkdims, addtags=ts"")
  dims = size(lattice)
  N = length(dims)
  linkinds_dict = Dict{Edge{N},Index{typeof(linkdims)}}()
  for n in sites(lattice), edge_n in incident_edges(lattice, n)
    l = Index(linkdims; tags=ITensors.addtags(link_tag(edge_n.edge...), addtags))
    get!(linkinds_dict, edge_n, l)
  end
  return linkinds_dict
end

function get_link_ind(linkinds_dict::Dict, edge::Edge, site::Tuple)
  l = linkinds_dict[edge]
  return is_in_edge(site, edge) ? dag(l) : l
end

# A network of link indices for a HyperCubic lattice, with
# no site indices.
function inds_network(dims::Int...; linkdims, kwargs...)
  site_inds = fill(Index{typeof(linkdims)}[], dims)
  return inds_network(site_inds; linkdims=linkdims, kwargs...)
end

function inds_network(site_inds::Array{<:Index,N}; kwargs...) where {N}
  return inds_network(map(x -> [x], site_inds); kwargs...)
end

# A network of link indices for a HyperCubic lattice, with
# site indices specified.
function inds_network(
  site_inds::Array{<:Vector{<:Index},N}; linkdims, addtags=ts"", periodic=true
) where {N}
  dims = size(site_inds)
  lattice = HyperCubic(dims)
  linkinds_dict = linkinds(lattice; linkdims=linkdims, addtags=addtags)
  inds = Array{Vector{Index{typeof(linkdims)}},N}(undef, dims)
  for n in sites(lattice)
    if periodic == true
      edges = incident_edges(lattice, n)
    else
      edges = [e for e in incident_edges(lattice, n) if e.boundary == false]
    end
    inds_n = [get_link_ind(linkinds_dict, edge_n, n) for edge_n in edges]
    inds[n...] = append!(inds_n, site_inds[n...])
  end
  return inds
end
