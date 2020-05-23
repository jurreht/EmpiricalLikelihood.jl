using Distributions
using EmpiricalLikelihood
using Random
using Test
using Statistics

Random.seed!(213)

het_moment(X, y, beta) = [y[1] - beta[1] - X[1] * beta[2]]

@testset "Heterskedasticity" for n in (50, 200)
    β = [1., 1.]
    β_test = zeros(500, 2)
    for rep in 1:500
        X = rand(LogNormal(), n)
        ϵ = rand(Normal(), n)
        u = ϵ .* @. sqrt(.1 .+ .2 * X .+ .3 * X.^2)
        y = β[1] .+ X * β[2] .+ u
        β_test[rep, :] = optimize_el(1, het_moment, reshape(X, n, 1), reshape(y, n, 1), β, .7622)[1]
    end
    @test mean(β_test, dims=1)[1, :] ≈ β
end
