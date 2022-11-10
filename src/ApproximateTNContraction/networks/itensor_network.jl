"""
    boundary_projectors(tn::Matrix{ITensor}, state=1)
For a 2D tensor network, return the right and bottom boundary projectors onto
the local state `state`.
"""
function boundary_projectors(tn::Matrix{ITensor}, state=1)
  top_row = tn[1, :]
  bottom_row = tn[end, :]
  left_column = tn[:, 1]
  right_column = tn[:, end]
  bottom_boundary_inds = commonind.(bottom_row, top_row)
  right_boundary_inds = commonind.(right_column, left_column)
  ψr = ITensors.state.(right_boundary_inds, state)
  ψb = ITensors.state.(bottom_boundary_inds, state)
  return ψr, ψb
end

function boundary_projectors(tn::Array{ITensor,3}, state=1)
  top = tn[1, :, :]
  bottom = tn[end, :, :]
  left = tn[:, 1, :]
  right = tn[:, end, :]
  front = tn[:, :, 1]
  back = tn[:, :, end]
  top_inds = commonind.(top, bottom)
  left_inds = commonind.(left, right)
  front_inds = commonind.(front, back)

  psi_top = ITensors.state.(top_inds, state)
  psi_left = ITensors.state.(left_inds, state)
  psi_front = ITensors.state.(front_inds, state)
  return psi_top, psi_left, psi_front
end

"""
    project_boundary(tn::Matrix{ITensor}, state=1)
Project the boundary of a periodic 2D tensor network onto 
the specified state.
"""
function project_boundary(tn::Matrix{ITensor}, state=1)
  Nx, Ny = size(tn)
  ψr, ψb = boundary_projectors(tn, state)
  for n in 1:Nx
    tn[n, 1] = tn[n, 1] * ψr[n]
    tn[n, end] = tn[n, end] * dag(ψr[n])
  end
  for n in 1:Ny
    tn[1, n] = tn[1, n] * ψb[n]
    tn[end, n] = tn[end, n] * dag(ψb[n])
  end
  return tn
end

function project_boundary(tn::Array{ITensor,3}, state=1)
  Nx, Ny, Nz = size(tn)
  psi_top, psi_left, psi_front = boundary_projectors(tn, state)
  for j in 1:Ny
    for k in 1:Nz
      tn[1, j, k] = tn[1, j, k] * psi_top[j, k]
      tn[end, j, k] = tn[end, j, k] * psi_top[j, k]
    end
  end
  for i in 1:Nx
    for k in 1:Nz
      tn[i, 1, k] = tn[i, 1, k] * psi_left[i, k]
      tn[i, end, k] = tn[i, end, k] * psi_left[i, k]
    end
  end
  for i in 1:Nx
    for j in 1:Ny
      tn[i, j, 1] = tn[i, j, 1] * psi_front[i, j]
      tn[i, j, end] = tn[i, j, end] * psi_front[i, j]
    end
  end
  return tn
end

function ITensors.prime(indices::Array{<:Index,1}, network::Array{ITensor}, n::Integer=1)
  function primeinds(tensor)
    prime_inds = [ind for ind in inds(tensor) if ind in indices]
    if (length(prime_inds) == 0)
      return tensor
    end
    return replaceinds(tensor, prime_inds => prime(prime_inds, n))
  end
  return map(x -> primeinds(x), network)
end

function ITensors.replaceinds(
  network::Union{Array{ITensor},Array{OrthogonalITensor}}, sim_dict::Dict
)
  if length(network) == 0
    return network
  end
  indices = collect(keys(sim_dict))
  function siminds(tensor)
    sim_inds = [ind for ind in inds(tensor) if ind in indices]
    if (length(sim_inds) == 0)
      return tensor
    end
    outinds = map(i -> sim_dict[i], sim_inds)
    return replaceinds(tensor, sim_inds => outinds)
  end
  return map(x -> siminds(x), network)
end

function ITensors.commoninds(n1::Array{ITensor}, n2::Array{ITensor})
  return mapreduce(a -> commoninds(a...), vcat, zip(n1, n2))
end
