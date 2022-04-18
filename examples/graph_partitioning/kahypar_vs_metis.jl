using Graphs
using KaHyPar
using Metis
using Suppressor

g = grid((16,))
npartitions = 2

configs = readdir(joinpath(pkgdir(KaHyPar), "src", "config"))
display(configs)

#configuration = :connectivity
configuration = :edge_cut
#configuration = joinpath(pkgdir(KaHyPar), "src", "config", "cut_kKaHyPar_sea20.ini")
#configuration = joinpath(pkgdir(KaHyPar), "src", "config", "km1_kKaHyPar_sea20.ini")

h = KaHyPar.hypergraph(adjacency_matrix(g))

# Make edge weights negative, helps a little
h.e_weights .*= -1

kahypar_partitions = @suppress KaHyPar.partition(h, npartitions; configuration)
metis_partitions = Metis.partition(g, npartitions)
@show kahypar_partitions
@show metis_partitions
