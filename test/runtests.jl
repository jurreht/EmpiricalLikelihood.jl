using Distributions
using EmpiricalLikelihood
using Ipopt
using JuMP
using Random
using Test
using Statistics

het_moment(X, y, beta) = [y[1] - beta[1] - X[1] * beta[2]]

@testset "Heteroskedasticity" begin
    n = 50
    Random.seed!(213)
    β = [1., 1.]
    β_test = zeros(100, 2)
    for rep in 1:100
        X = rand(LogNormal(), n)
        ϵ = rand(Normal(), n)
        u = ϵ .* @. sqrt(.1 .+ .2 * X .+ .3 * X.^2)
        y = β[1] .+ X * β[2] .+ u
        β_test[rep, :] = optimize_el(1, het_moment, reshape(X, n, 1), reshape(y, n, 1), β, .35)[1]
    end
    @test mean(β_test, dims=1)[1, :] ≈ β atol=1e-1
end

@testset "Test JuMP interface" begin
    n = 50
    Random.seed!(213)
    β = [1., 1.]
    β_test = zeros(100, 2)
    for rep in 1:100
        X = rand(LogNormal(), n)
        ϵ = rand(Normal(), n)
        u = ϵ .* @. sqrt(.1 .+ .2 * X .+ .3 * X.^2)
        y = β[1] .+ X * β[2] .+ u
        model = setup_jump_problem(1, het_moment, reshape(X, n, 1), reshape(y, n, 1), β, .35)[1]
        set_optimizer(model, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
        optimize!(model)
        β_test[rep, :] = @. value(model[:theta])
    end
    @test mean(β_test, dims=1)[1, :] ≈ β atol=1e-1
end
