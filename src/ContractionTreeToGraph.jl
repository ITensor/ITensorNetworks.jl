
"""The main object here is `g' a Datagraph which represents a graphical version of a contraction sequence.
It's vertices describe a partition between the leaves of the sequence (will be labelled with an n = 1 or n = 3 element tuple, where each element of the tuple describes the leaves in one of those partition)
n = 1 implies the vertex is actually a leaf.
Edges connect vertices which are child/ parent"""


"""Function to take a sequence (returned by ITensorNetworks.contraction_sequence) and construct a graph g which represents it (see above)"""
function create_contraction_tree(contract_sequence)
    #Collect all the leaves
    leaves = collect(Leaves(contract_sequence))
    #Build empty graph
    g = NamedGraph()
    #Construct the left branch
    spawn_child_branch(contract_sequence[1], g, leaves)
    #Construct the right branch
    spawn_child_branch(contract_sequence[2], g, leaves)

    #Now we have the vertices we need to figure out the edges
    for v in vertices(g)
        #Only add edges from a parent (which defines a tripartition and thus has length 3) to its children
        if(length(v) == 3)
        #Work out which vertices it connects to
            concat1, concat2, concat3 =[v[1]..., v[2]...], [v[2]..., v[3]...], [v[1]..., v[3]...]
            for vn in setdiff(vertices(g), [v])
                vn_set = [Set(vni) for vni in vn]
                if(Set(concat1) ∈ vn_set || Set(concat2) ∈ vn_set || Set(concat3) ∈ vn_set)
                    add_edge!(g, v => vn)
                end
            end
        end
    end


    return g
end

"""Given a contraction sequence which is a subsequence of some larger sequence which is being built on current_g and has leaves `leaves`
Spawn `contract sequence' as a vertex on `current_g' and continue on with its children """
function spawn_child_branch(contract_sequence, current_g, leaves)
    #Check if sequence is at leaf point
    if(isa(contract_sequence, Array))
        #If not it  will spawn two children
        group1 = collect(Leaves(contract_sequence[1]))
        group2 = collect(Leaves(contract_sequence[2]))
        remaining_verts = setdiff(leaves, vcat(group1, group2))
        add_vertex!(current_g, (group1, group2, remaining_verts))
        spawn_child_branch(contract_sequence[1], current_g, leaves)
        spawn_child_branch(contract_sequence[2], current_g, leaves)
    else
        #If it is it is just a vertex
        add_vertex!(current_g, ([contract_sequence], setdiff(leaves, [contract_sequence])))
    end
end

"""Utility functions for the graphical representation of a contraction sequence
Perhaps it should be a specific object or type of tree?!"""

"""Determine if a node is a leaf"""
function leaf_node(g::AbstractGraph, v)
    return length(neighbors(g, v)) == 1
end

"""Determine if an edge involves a leaf (at src or dst)"""
function leaf_edge(g::AbstractGraph, e)
    return leaf_node(g, src(e)) || leaf_node(g, dst(e))
end 

"""Determine if a node has no neighbours which are leaves"""
function no_leaf_neighbours(g::AbstractGraph, v)
    for vn in neighbors(g, v)
        if(leaf_node(g, vn))
            return false
        end
    end
    return true
end

"""Get all edges which do not involve a leaf
(why does edges(g)[contraction_edge(g, edges(g)) .== true] not work?)"""
function non_leaf_edges(g::AbstractGraph)
    return edges(g)[findall(==(1), [!leaf_edge(g,e) for e in edges(g)])]
end

"""Get all nodes which aren't leaves and don't have leaf neighbours (these represent internal points in the contraction sequence)"""
function internal_nodes(g::AbstractGraph)
    return vertices(g)[findall(==(1), [no_leaf_neighbours(g, v) && !leaf_node(g, v) for v in vertices(g)])]
end

"""Get all nodes which have leaf neighbours but aren't leaves themselves (these represent start/ end points of a contraction sequence)"""
function external_nodes(g::AbstractGraph)
    return vertices(g)[findall(==(1), [!no_leaf_neighbours(g, v) && !leaf_node(g,v)  for v in vertices(g)])]
end

"""Get the vertex bi-partition that a given edge between non-leaf nodes represents"""
function edge_bipartition(g::AbstractGraph, e)

    if(leaf_edge(g, e))
        println("ERROR: EITHER THE SOURCE OR THE VERTEX IS A LEAF SO EDGE DOESN'T REALLY REPRESENT A BI-PARTITION")
    end

    vsrc_set, vdst_set = [Set(vni) for vni in src(e)], [Set(vni) for vni in dst(e)]
    c1, c2, c3 = [src(e)[1]..., src(e)[2]...], [src(e)[2]..., src(e)[3]...], [src(e)[1]..., src(e)[3]...]
    left_bipartition = Set(c1) ∈ vdst_set ? c1 : Set(c2) ∈ vdst_set ? c2 : c3

    c1, c2, c3 = [dst(e)[1]..., dst(e)[2]...], [dst(e)[2]..., dst(e)[3]...], [dst(e)[1]..., dst(e)[3]...]
    right_bipartition = Set(c1) ∈ vsrc_set ? c1 : Set(c2) ∈ vsrc_set ? c2 : c3

    return left_bipartition, right_bipartition
end

"""Given a contraction node, get the keys from all its neighbouring leaves"""
function external_node_keys(g::AbstractGraph, v)
    return [Base.Iterators.flatten(v[findall(==(1), [length(vi) == 1 for vi in v])])...]
end

"""Given a contraction node, get all keys which are not from a neighbouring leaf"""
function external_contraction_node_ext_keys(g::AbstractGraph, v)
    return [Base.Iterators.flatten(v[findall(==(1), [length(vi) != 1 for vi in v])])...]
end
