# -*- coding: utf-8 -*-
# run_gaussdca.jl
# author : Antoine Passemiers

using GaussDCA


DATA_FOLDER = "../data/training_set"

for (root, dirs, files) in walkdir(DATA_FOLDER)
    for file in files
        if (file == "trimmed.a3m") || (file == "sequence.fa.blits4.trimmed")
            println(joinpath(root, file))
            if !isfile(joinpath(root, "dir.gaussdca"))
                DIR = gDCA(joinpath(root, file), score = :DI, min_separation = 1)
                printrank(joinpath(root, "dir.gaussdca"), DIR)
            end
        end
    end
end