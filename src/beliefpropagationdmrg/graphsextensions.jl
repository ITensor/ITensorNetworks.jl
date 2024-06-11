using Graphs: AbstractGraph
using NamedGraphs.GraphsExtensions: a_star, add_edge!, dfs_tree, post_order_dfs_edges
using NamedGraphs.GraphsExtensions: default_root_vertex, degree, degrees, neighbors, edgetype, rem_edge!, vertextype

#Given a graph, traverse it from start vertex to end vertex, covering each edge exactly once.
#Complexity is O(length(edges(g)))
function eulerian_path(g::AbstractGraph, start_vertex, end_vertex)
    #Conditions on g for the required path to exist
    if start_vertex != end_vertex
        @assert isodd(degree(g, start_vertex) % 2)
        @assert isodd(degree(g, end_vertex) % 2)
        @assert all(x -> iseven(x), degrees(g, setdiff(vertices(g), [start_vertex, end_vertex])))
    else
        @assert all(x -> iseven(x), degrees(g))
    end

    path = vertextype(g)[]
    stack = vertextype(g)[]
    current_vertex = end_vertex
    g_modified = copy(g)
    while !isempty(stack) || !iszero(degree(g_modified, current_vertex))
        if iszero(degree(g_modified, current_vertex))
            push!(path, current_vertex)
            last_vertex = pop!(stack)
            current_vertex = last_vertex
        else
            push!(stack, current_vertex)
            vn = first(neighbors(g_modified, current_vertex))
            rem_edge!(g_modified, edgetype(g_modified)(current_vertex, vn))
            current_vertex = vn
        end
    end
    
    push!(path, current_vertex)

    return edgetype(g_modified)[edgetype(g_modified)(path[i], path[i+1]) for i in 1:(length(path) - 1)] 
end

eulerian_cycle(g::AbstractGraph, start_vertex) = eulerian_path(g, start_vertex, start_vertex)

function make_all_degrees_even(g::AbstractGraph)
    g_modified = copy(g)
    vertices_odd_degree = collect(filter(v -> isodd(degree(g, v)), vertices(g_modified)))
    while !isempty(vertices_odd_degree)
        vertex_pairs = [(vertices_odd_degree[i], vertices_odd_degree[j]) for i in 1:length(vertices_odd_degree) for j in (i+1):length(vertices_odd_degree)]
        vertex_pair = first(sort(vertex_pairs; by = vp -> length(a_star(g, vp...))))
        add_edge!(g_modified, edgetype(g_modified)(vertex_pair...))
        vertices_odd_degree = filter(v -> v != first(vertex_pair) && v != last(vertex_pair), vertices_odd_degree)
    end
    return g_modified
end

#Given a graph, traverse it in a cycle from start_vertex and try to minimise the number of edges traversed more than once
function _eulerian_cycle(g::AbstractGraph, start_vertex; add_additional_traversal = false)
    g_modified = make_all_degrees_even(g::AbstractGraph)
    path = eulerian_cycle(g_modified, start_vertex)

    !add_additional_traversal && return path

    modified_path = edgetype(g_modified)[]

    for e in path
        if src(e) ∉ neighbors(g, dst(e))
            inner_path = a_star(g, src(e), dst(e))
            append!(modified_path, inner_path)
        else
            push!(modified_path, e)
        end
    end

    return modified_path
end

function _bp_region_plan(g::AbstractGraph, start_vertex = default_root_vertex(g); nsites::Int = 1, add_additional_traversal = false)
    path = _eulerian_cycle(g, start_vertex; add_additional_traversal)
    if nsites == 1
        regions = [[v] for v in vcat(src.(path))]
        @assert all( v -> only(v) ∈ vertices(g), regions)
        return vcat(regions, reverse(regions))
    else
        regions = filter(e -> e ∈ edges(g) || reverse(e) ∈ edges(g), path)
        @assert all(e -> e ∈ regions || reverse(e) ∈ regions, edges(g))
        return regions
    end
end

function path_to_path(path)
    verts = []
    for e in path
        if isempty(verts) || src(e) ≠ last(verts)
            push!(verts, src(e))
            push!(verts, dst(e))
        end
    end
    return [[v] for v in verts]
end

function bp_region_plan(g::AbstractGraph, start_vertex = default_root_vertex(g); nsites::Int = 1, add_additional_traversal = false)
    path = post_order_dfs_edges(g, start_vertex)
    if nsites == 1
        regions = path_to_path(path)
        @assert all( v -> only(v) ∈ vertices(g), regions)
        return vcat(regions, reverse(regions))
    else
        regions = filter(e -> e ∈ edges(g) || reverse(e) ∈ edges(g), path)
        @assert all(e -> e ∈ regions || reverse(e) ∈ regions, edges(g))
        return regions
    end
end