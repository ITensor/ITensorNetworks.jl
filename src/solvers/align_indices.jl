
function align_indices(tn)
  si = siteinds(tn)
  ptn = copy(tn)
  for v in vertices(tn)
    is = inds(tn[v])
    ls = setdiff(is, si[v])
    isempty(ls) && continue
    new_is = [first(ls), si[v]...]
    if length(ls) >= 2
      new_is = vcat(new_is, ls[2:end])
    end
    ptn[v] = permute(tn[v], new_is)
  end
  return ptn
end
