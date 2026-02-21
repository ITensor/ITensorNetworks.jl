using Literate: Literate

let inputfile = joinpath(@__DIR__, "..", "examples", "README.jl"),
        outputdir = joinpath(@__DIR__, ".."), flavor = Literate.CommonMarkFlavor(),
        name = "README"

    function postprocess(content)
        include_ccq_logo = """
        <picture>
          <source media="(prefers-color-scheme: dark)" width="20%" srcset="docs/src/assets/CCQ-dark.png">
          <img alt="Flatiron Center for Computational Quantum Physics logo." width="20%" src="docs/src/assets/CCQ.png">
        </picture>
        """
        return replace(content, "{CCQ_LOGO}" => include_ccq_logo)
    end
    Literate.markdown(inputfile, outputdir; flavor, name, postprocess)
end
