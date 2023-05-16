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

"""Given a vector of ITensors, separate them into groups of commuting itensors (i.e. itensors in the same group do not share any common indices)"""
function group_ITensors(its::Vector{ITensor})
  remaining_its = copy(its)
  it_groups = Vector{ITensor}[]

  while !isempty(remaining_its)
    cur_group = ITensor[]
    cur_indices = Index[]
    inds_to_remove = []
    for i in 1:length(remaining_its)
      it = remaining_its[i]
      it_inds = inds(it)

      if all([i ∉ cur_indices for i in it_inds])
        push!(cur_group, it)
        push!(cur_indices, it_inds...)
        push!(inds_to_remove, i)
      end
    end
    remaining_its = ITensor[
      remaining_its[i] for
      i in setdiff([i for i in 1:length(remaining_its)], inds_to_remove)
    ]
    push!(it_groups, cur_group)
  end

  return it_groups
end
