using Dictionaries: Dictionary
using Graphs: AbstractGraph, nv, vertices
using ITensors: ITensors, Index, siteind

function ITensors.siteind(sitetype::String, v::Tuple; kwargs...)
    return addtags(siteind(sitetype; kwargs...), vertex_tag(v))
end

# naming collision of ITensors.addtags and addtags keyword in siteind system
function ITensors.siteind(d::Integer, v; addtags = "", kwargs...)
    return ITensors.addtags(Index(d; tags = "Site, $addtags", kwargs...), vertex_tag(v))
end

to_siteinds_callable(x) = Returns(x)
function to_siteinds_callable(x::AbstractDictionary)
    return Base.Fix1(getindex, x) ∘ keytype(x)
end

function siteinds(x, g::AbstractGraph; kwargs...)
    return siteinds(to_siteinds_callable(x), g; kwargs...)
end

# Convenient syntax for path graph.
function siteinds(x, nv::Int; kwargs...)
    return siteinds(x, path_graph(nv); kwargs...)
end

function to_siteind(x, vertex; kwargs...)
    return [siteind(x, vertex_tag(vertex); kwargs...)]
end

to_siteind(x::Index, vertex; kwargs...) = [x]

function siteinds(f::Function, g::AbstractGraph; kwargs...)
    is = IndsNetwork(g)
    for v in vertices(g)
        is[v] = to_siteind(f(v), v; kwargs...)
    end
    return is
end
