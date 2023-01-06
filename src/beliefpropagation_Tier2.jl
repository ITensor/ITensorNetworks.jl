#THESE ARE HIGHER LEVEL FUNCTIONS FOR BELIEF PROPAGATION

#Given an ITensorNetwork tn with site indices, message tensors and a subset of vertices in tn 
#Compute the reduced density matrix on the subset of vertices
#Function currenly assumes the message tensors have been formulated over tn ⊗ prime(dag(tn); sites = []).
function calculate_reduced_density_matrix_BP(tn::ITensorNetwork, mts::DataGraph, verts::Vector)
  tntn = tn ⊗ prime(dag(tn); sites = verts)
  verts_for_env = vcat([(v,1) for v in verts],[(v,2) for v in verts])
  environment_tensors = get_environment_BP(tntn, mts, verts_for_env)
  envs = flatten([environment_tensors[i] for i in keys(environment_tensors)])
  rdm = calculate_contraction_BP(tntn, mts, verts_for_env, [tntn[v] for v in verts_for_env])

  return rdm

end

#Function to calculate calculate_reduced_density_matrix_BP, formulating the message tensors first.
#Network is assumed to have open site indices.
function calculate_reduced_density_matrix_BP(tn::ITensorNetwork, verts::Vector; nvertices_per_partition=1, niters=10, vertex_groups = nothing)
  tntn = tn ⊗ prime(dag(tn); sites = [])
  
  mts = compute_message_tensors(tntn; nvertices_per_partition=nvertices_per_partition, niters=niters, vertex_groups = vertex_groups)

  return calculate_reduced_density_matrix_BP(tn, mts, verts)

end