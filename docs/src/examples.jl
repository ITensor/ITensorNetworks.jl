# # Examples

using Graphs: path_graph
using ITensorNetworks: ITensorNetwork
tn = ITensorNetwork(path_graph(4); link_space=2)
