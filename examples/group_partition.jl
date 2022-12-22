using ITensors
using Graphs
using NamedGraphs
using ITensorNetworks
using SplitApplyCombine
using Metis

s = siteinds("S=1/2", named_grid(8))
tn = ITensorNetwork(s; link_space=2)
Z = prime(tn; sites=[]) ⊗ tn
vertex_groups = group(v -> v[1], vertices(Z))
# Create two layers of partitioning
Z_p = partition(partition(Z, vertex_groups); nvertices_per_partition=2)
# Flatten the partitioned partitions
Z_verts = [reduce(vcat, (vertices(Z_p[vp][v]) for v in vertices(Z_p[vp]))) for vp in vertices(Z_p)]
