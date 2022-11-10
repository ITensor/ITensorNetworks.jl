mutable struct OrthogonalITensor
  tensor::ITensor
  ortho_indices::Vector
end

function OrthogonalITensor(tensor::ITensor)
  return OrthogonalITensor(tensor, [])
end

function orthogonal_tensors(tensors::Vector{ITensor})
  return [OrthogonalITensor(t) for t in tensors]
end

function get_tensors(ortho_tensors::Vector{OrthogonalITensor})
  return [t.tensor for t in ortho_tensors]
end

function ITensors.noncommoninds(ortho_tensors::OrthogonalITensor...)
  if length(ortho_tensors) == 1
    return collect(inds(ortho_tensors[1]))
  end
  return noncommoninds(get_tensors([ortho_tensors...])...)
end

function ITensors.inds(ortho_tensor::OrthogonalITensor)
  return inds(ortho_tensor.tensor)
end

function ITensors.replaceinds(ortho_tensor::OrthogonalITensor, change_inds::Pair)
  input_inds = change_inds[1]
  output_inds = change_inds[2]
  input_to_output = Dict{Index,Index}()
  for (i_in, i_out) in zip(input_inds, output_inds)
    input_to_output[i_in] = i_out
  end
  new_ortho_indices = []
  for i in ortho_tensor.ortho_indices
    if haskey(input_to_output, i)
      push!(new_ortho_indices, input_to_output[i])
    else
      push!(new_ortho_indices, i)
    end
  end
  new_tensor = replaceinds(ortho_tensor.tensor, change_inds)
  return OrthogonalITensor(new_tensor, new_ortho_indices)
end
