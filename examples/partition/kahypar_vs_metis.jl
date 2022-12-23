using Graphs
using KaHyPar
using Metis
using ITensorNetworks

g = grid((16,))
npartitions = 4

kahypar_partitions = subgraph_vertices(g; npartitions, backend="KaHyPar")
metis_partitions = subgraph_vertices(g; npartitions, backend="Metis")
@show kahypar_partitions, length(kahypar_partitions)
@show metis_partitions, length(metis_partitions)

g_parts = partition(g; npartitions)
@show nv(g_parts) == 4
@show nv(g_parts[1]) == 4
@show nv(g_parts[2]) == 4
@show nv(g_parts[3]) == 4
@show nv(g_parts[4]) == 4
@show issetequal(metis_partitions[2], vertices(g_parts[2]))

using ITensorNetworks
tn = ITensorNetwork(named_grid((4, 2)); link_space=3);

# subgraph_vertices
tn_sv = subgraph_vertices(tn; npartitions=2) # Same as `partition_vertices(tn; nvertices_per_partition=4)`

# partition_vertices
tn_pv = partition_vertices(tn; npartitions=2);
typeof(tn_pv)
tn_pv[1]
edges(tn_pv)
tn_pv[1 => 2]

# subgraphs
tn_sg = subgraphs(tn; npartitions=2);
typeof(tn_sg)
tn_sg[1]

# partition
tn_pg = partition(tn; npartitions=2);
typeof(tn_pg)
tn_pg[1]
tn_pg[1 => 2][:edges]
