using Documenter
using MPIMapReduce

DocMeta.setdocmeta!(MPIMapReduce, :DocTestSetup, :(using MPIMapReduce); recursive=true)

makedocs(;
    modules=[MPIMapReduce],
    authors="Jishnu Bhattacharya",
    repo="https://github.com/jishnub/MPIMapReduce.jl/blob/{commit}{path}#L{line}",
    sitename="MPIMapReduce.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jishnub.github.io/MPIMapReduce.jl",
        assets=String[],
    ),
    pages=[
        "MPIMapReduce" => "index.md",
        "Examples" => "examples.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/jishnub/MPIMapReduce.jl",
)
