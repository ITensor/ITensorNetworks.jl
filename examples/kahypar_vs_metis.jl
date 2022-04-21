using ITensorNetworks

g = grid((16,))
npartitions = 2

kahypar_partitions = partition(g, npartitions; backend="KaHyPar")
metis_partitions = partition(g, npartitions; backend="Metis")
@show kahypar_partitions
@show metis_partitions
