using LightGraphs: LightGraphs
using KaHyPar
using Metis
using Suppressor

g = LightGraphs.grid((16,))
npartitions = 4

kahypar_partitions = @suppress KaHyPar.partition(adjacency_matrix(g), npartitions)
metis_partitions = Metis.partition(g, npartitions)

@show kahypar_partitions
@show metis_partitions
