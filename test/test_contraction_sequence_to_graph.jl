using ITensorNetworks
using ITensorNetworks: contraction_sequence_to_graph, leaf_node, non_leaf_edges, edge_bipartition
using Test
using ITensors

@testset "contraction_sequence_to_graph" begin

    n =3
    dims = (n,n)
    g = named_grid(dims)
    s = siteinds("S=1/2", g)

    ψ = randomITensorNetwork(s; link_space=2)
    ψψ = flatten_networks(ψ,ψ)

    seq = contraction_sequence(ψψ);

    g_seq = contraction_sequence_to_graph(seq)

    #Get all leaf nodes (should match number of tensors in original network)
    #why does vertices(g_seq)[leaf_node(g_seq, vertices(g_seq))) .== true] not work?"""
    g_seq_leaves = vertices(g_seq)[findall(==(1), [leaf_node(g_seq,v) for v in vertices(g_seq)])]

    @test length(g_seq_leaves) == n*n

    g_seq_internal_edges = non_leaf_edges(g_seq)

    internal_edge_test = 0
    #Check all internal edges define a correct bi-partition
    for eb in g_seq_internal_edges
      es = edge_bipartition(g_seq, eb)
      if(length(es) != 2)
        internal_edge_test += 1
      end

      es =  vcat(es[1],es[2])

      if(Set([e.I for e in es]) != Set(vertices(ψψ)))
        internal_edge_test += 1
      end
    end

    @test internal_edge_test == 0

    #Check all internal vertices define a correct tripartition and all leaf vertices define a bipartition (tensor on that leafs vs tensor on rest of tree)
    vertex_test = 0
    for v in vertices(g_seq)
      if(!leaf_node(g_seq, v))
        if(length(v) != 3)
          vertex_test += 1
        end
        vs =  vcat(v[1], v[2], v[3])
        if(Set([vsi.I for vsi in vs]) != Set(vertices(ψψ)))
          vertex_test += 1
        end
      else
        if(length(v) != 2)
          vertex_test += 1
        end
        vs =  vcat(v[1], v[2])
        if(Set([vsi.I for vsi in vs]) != Set(vertices(ψψ)))
          vertex_test += 1
        end
      end

    end

    @test vertex_test == 0
    
  end