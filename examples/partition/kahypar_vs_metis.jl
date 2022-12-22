using Graphs
using KaHyPar
using Metis
using ITensorNetworks

g = grid((16,))
npartitions = 4

kahypar_partitions = partition_vertices(g; npartitions, backend="KaHyPar")
metis_partitions = partition_vertices(g; npartitions, backend="Metis")
@show kahypar_partitions, length(kahypar_partitions)
@show metis_partitions, length(metis_partitions)

g_parts = partition(g; npartitions)
@show nv(g_parts) == 4
@show nv(g_parts[1]) == 4
@show nv(g_parts[2]) == 4
@show nv(g_parts[3]) == 4
@show nv(g_parts[4]) == 4
@show issetequal(metis_partitions[2], vertices(g_parts[2]))
