# -*- coding: utf-8 -*-
# run_gaussdca.jl
# author : Antoine Passemiers

using GaussDCA


for (root, dirs, files) in walkdir(".")
    for file in files
        if (file == "alignment.a3m") || (file == "sequence.fa.blits4.trimmed")
            println(joinpath(root, file))
            if !isfile(joinpath(root, "fnr.gaussdca"))
                FNR = gDCA(joinpath(root, file), score = :frob, min_separation = 1)
                printrank(joinpath(root, "fnr.gaussdca"), FNR)
            end
            if !isfile(joinpath(root, "dir.gaussdca"))
                DIR = gDCA(joinpath(root, file), score = :DI, min_separation = 1)
                printrank(joinpath(root, "dir.gaussdca"), DIR)
            end
        end
    end
end