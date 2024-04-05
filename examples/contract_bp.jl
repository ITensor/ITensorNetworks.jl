# https://docs.julialang.org/en/v1/base/base/#Base.names
function imported_names(M::Module)
  return setdiff(names(M; imported=true), names(M; imported=false))
end

function imported_functions(M::Module)
  return map(f -> getfield(M, f), imported_names(M))
end

using FileUtils: replace_in_files

function prepend_imports(M::Module)
  fs = imported_functions(M)
  # foreach(f -> @show((parentmodule(f), f)), fs)
  src_path = joinpath(pkgdir(M), "src")
  @show src_path
  for f in fs
    function_name = last(split(string(f), "."))
    module_name = last(split(string(parentmodule(f)), "."))
    annotated_function_name = module_name * "." * function_name
    @show function_name, annotated_function_name
    replace_in_files(
      src_path,
      "function $(function_name)(" => "function $(annotated_function_name)(";
      recursive=true,
      ignore_dirs=[".git"],
      showdiffs=false,
    )
    replace_in_files(
      src_path,
      "\n$(function_name)(" => "\n$(annotated_function_name)(";
      recursive=true,
      ignore_dirs=[".git"],
      showdiffs=false,
    )
  end
end

using ITensorNetworks: ITensorNetworks
prepend_imports(ITensorNetworks)
