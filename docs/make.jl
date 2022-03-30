using RobustTDA
using Documenter

DocMeta.setdocmeta!(RobustTDA, :DocTestSetup, :(using RobustTDA); recursive=true)

makedocs(;
    modules=[RobustTDA],
    authors="Siddharth Vishwanath",
    repo="https://github.com/sidv23/RobustTDA.jl/blob/{commit}{path}#{line}",
    sitename="RobustTDA.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sidv23.github.io/RobustTDA.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/sidv23/RobustTDA.jl",
    devbranch="main",
)
