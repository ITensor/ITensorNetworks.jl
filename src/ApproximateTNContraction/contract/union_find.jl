struct UF
  parent_map::Dict
end

function UF(values::Vector)
  parent_map = Dict()
  for value in values
    parent_map[value] = value
  end
  return UF(parent_map)
end

function root(uf::UF, n)
  while uf.parent_map[n] != n
    n = uf.parent_map[n]
  end
  return n
end

function connect(uf, n1, n2)
  rootn1 = root(uf, n1)
  rootn2 = root(uf, n2)
  if rootn1 == rootn2
    # Already connected
    return nothing
  end
  return uf.parent_map[rootn1] = rootn2
end
