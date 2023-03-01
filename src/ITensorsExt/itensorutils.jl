using ITensors.NDTensors: Tensor, diaglength, getdiagindex, setdiagindex!, tensor

function map_diag!(f::Function, it_destination::ITensor, it_source::ITensor)
  return itensor(map_diag!(f, tensor(it_destination), tensor(it_source)))
end
map_diag(f::Function, it::ITensor) = map_diag!(f, copy(it), it)

function map_diag!(f::Function, t_destination::Tensor, t_source::Tensor)
  for i in 1:diaglength(t_destination)
    setdiagindex!(t_destination, f(getdiagindex(t_source, i)), i)
  end
  return t_destination
end
map_diag(f::Function, t::Tensor) = map_diag!(f, copy(t), t)

# Convenience functions
sqrt_diag(it::ITensor) = map_diag(sqrt, it)
inv_diag(it::ITensor) = map_diag(inv, it)
invsqrt_diag(it::ITensor) = map_diag(inv ∘ sqrt, it)
pinv_diag(it::ITensor) = map_diag(pinv, it)
pinvsqrt_diag(it::ITensor) = map_diag(pinv ∘ sqrt, it)
