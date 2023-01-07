#THESE ARE HIGHER LEVEL FUNCTIONS FOR CALCULATING PHYSICAL PROPERTIES OF TNS (RDMS, BITSTRINGS, EXPEC VALUES)

#Given an ITensorNetwork tn = psi ⊗ prime(dag(psi); sites = []), message tensors formulated over it and a subset of vertices in psi
#Compute the reduced density matrix on the subset of vertices
function calculate_reduced_density_matrix(tn::ITensorNetwork, mts::DataGraph, verts::Vector)
  vbra, vket = [(v,1) for v in verts],[(v,2) for v in verts]
  rdm = calculate_contraction(tn, mts, vcat(vbra, vket), vcat([tn[v] for v in vbra],[prime(tn[v], tags="Site") for v in vket]))

  return rdm

end