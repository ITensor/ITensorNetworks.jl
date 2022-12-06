using Graphs
using KaHyPar
using Metis
using ITensorNetworks

g = grid((16,))
npartitions = 4

kahypar_partitions = partition(g, npartitions; backend="KaHyPar")
metis_partitions = partition(g, npartitions; backend="Metis")
@show kahypar_partitions, length(unique(kahypar_partitions))
@show metis_partitions, length(unique(metis_partitions))
