module EmpiricalLikelihood

using DiffResults
using ForwardDiff
using JuMP
using Optim
using PyCall
using Statistics

include("HigherOrderKernels.jl")

const scipy_spatial = PyNULL()

function __init__()
    copy!(scipy_spatial, pyimport_conda("scipy.spatial", "scipy"))
end

export optimize_el, setup_jump_problem

const Kernel = EpanechnikovKernel{2}

function optimize_el(n_moments::Integer, moment_func, X::Matrix{T}, Y::Matrix{T}, start_param::Vector{T})::Tuple{Vector{T}, T} where T <: AbstractFloat
    # Automatic bandwith selection according to Silverman's rule of thumb.
    # This is not the best we can do, e.g. in the paper it is suggested to use
    # cross-validation a la Newey (1993). However, since according to the paper
    # the bandwith is not very important, we just use something simple.
    # So we take the average bandwith over the bandwith's suggested
    b = mean(density_bandwidth(Kernel, X[:, i]) for i in 1:size(X, 2))
    return optimize_el(n_moments, moment_func, X, Y, start_param, b)
end

function optimize_el(n_moments::Int, moment_func, X::Matrix{T}, Y::Matrix{T}, start_param::Vector{T}, bandwidth::T, trimming::T=.99999)::Tuple{Vector{T}, T} where T <: AbstractFloat
    return optimize_el(convert(UInt, n_moments), moment_func, X, Y, start_param, bandwidth, trimming)
end

function optimize_el(n_moments::UInt, moment_func, X::Matrix{T}, Y::Matrix{T}, start_param::Vector{T}, bandwidth::T, trimming::T=.99999)::Tuple{Vector{T}, T} where T <: AbstractFloat
    weights, not_trimmed = weights_and_trims(n_moments, moment_func, X, Y, start_param, bandwidth, trimming)

    res = Optim.optimize(
        Optim.only_fg!((f, g, x) -> el_objective!(n_moments, moment_func, X, Y, weights, not_trimmed, x, f, g)),
        start_param,
        LBFGS()
    )

    return Optim.minimizer(res), -1 * Optim.minimum(res)
end

function setup_jump_problem(n_moments::Int, moment_func, X::Matrix{T}, Y::Matrix{T}, start_param::Vector{T}, bandwidth::T, trimming::T=.99999)::Model where T <: AbstractFloat
    return setup_jump_problem(convert(UInt, n_moments), moment_func, X, Y, start_param, bandwidth, trimming)
end

function setup_jump_problem(n_moments::UInt, moment_func, X::Matrix{T}, Y::Matrix{T}, start_param::Vector{T}, bandwidth::T, trimming::T=.99999)::Model where T <: AbstractFloat
    weights, not_trimmed = weights_and_trims(n_moments, moment_func, X, Y, start_param, bandwidth, trimming)

    model = Model()
    n_params = length(start_param)
    @variable(model, theta[1:n_params])
    for i in 1:n_params
        set_start_value(theta[i], start_param[i])
    end
    neg_ll(x...) = begin
        x_vec = vcat(x...)
        el_objective!(n_moments, moment_func, X, Y, weights, not_trimmed, x_vec, 1, nothing)
    end
    ∇neg_ll(grad, x...) = begin
        x_vec = vcat(x...)
        el_objective!(n_moments, moment_func, X, Y, weights, not_trimmed, x_vec, nothing, grad)
    end
    register(model, :neg_empirical_likelihood, n_params, neg_ll, ∇neg_ll)
    @NLobjective(model, Min, neg_empirical_likelihood(theta...))

    return model
end

function weights_and_trims(n_moments::UInt, moment_func, X::Matrix{T}, Y::Matrix{T}, start_param::Vector{T}, bandwidth::T, trimming::T)::Tuple{Matrix{T}, Vector{Bool}} where T <: AbstractFloat
    # Calcualte weights
    n_obs, n_conditioning_vars = size(X)
    weights = Matrix{T}(undef, n_obs, n_obs)
    not_trimmed = Vector{Bool}(undef, n_obs)  # false = we will trim
    for i in 1:n_obs
        for j in 1:n_obs
            weights[i, j] = prod(
                density_kernel(Kernel, (X[i, k] - X[j, k]) / bandwidth)
                for k in 1:n_conditioning_vars
            )
        end

        # Determine if the observation will be trimmed.
        # Contrary to what is claimed in the paper, I find it impossible to get the
        # estimator even working without trimming. (Just consider the case where w_{ij} =1,
        # i.e. an observation without "neighbours". In these cases the inner loop is
        # always ill-defined.)
        # Nadaray-Watson estimator of pdf of X at x_i
        hhat = sum(weights[i, :]) / (n_obs * bandwidth^n_conditioning_vars)
        not_trimmed[i] = hhat >= bandwidth^trimming

        # Normalize weights
        weights[i, :] ./= sum(weights[i, :])
    end

    if !any(not_trimmed)
        throw("All observations are trimmed. Decrease the bandwith or increase the trimming parameter.")
    end

    # Check if starting paramater is valid
    moment_vals, _ = moment_vals_and_jac(n_moments, moment_func, X, Y, start_param)
    if !zero_in_hull_for_all(moment_vals, weights, not_trimmed)
        throw("Zero is not in convex hull of moments at starting value. Try a different starting value or increase the bandwidth.")
    end

    return weights, not_trimmed
end

function el_objective!(n_moments::UInt, moment_func, X::Matrix{S}, Y::Matrix{S}, weights::Matrix{S}, not_trimmed::Vector{Bool}, theta::Vector{T}, out, grad::Union{Nothing, AbstractVector{S}})::Union{Nothing, T} where {T <: Real, S <: AbstractFloat}
    moment_vals, moment_jac = moment_vals_and_jac(n_moments, moment_func, X, Y, theta)

    if !zero_in_hull_for_all(moment_vals, weights, not_trimmed)
        if !isnothing(grad)
            grad[:] .= NaN
        end
        if !isnothing(out)
            return -Inf
        end
        return
    end

    # Inner loop
    n_obs = size(X, 1)
    n_params = length(theta)
    λ = Matrix{T}(undef, n_obs, n_moments)
    # Obj contains the negative of the objective at the top of p. 1674, as we will minimize and not maximize
    obj = 0.
    if grad != nothing
        grad[:] = zeros(T, n_params)
    end
    for i in 1:n_obs
        if !not_trimmed[i]
            # This adds 0 to the SER, just go on
            continue
        end

        # Find λ maximizing as in (2.7) in the paper
        weight_i = weights[i, :]
        res = Optim.optimize(
            Optim.only_fgh!((f, g, h, γ) -> λ_obj_fgh(f, g, h, γ, moment_vals, weight_i)),
            # γ -> -1 * λ_obj(γ, moment_vals, weight_i),
            # (g, γ) -> -1 * λ_grad!(g, γ, moment_vals, weight_i),
            zeros(n_moments),
            Newton()
        )
        if !Optim.converged(res)
            throw("Inner loop did not converge")
        end
        λ[i, :] = Optim.minimizer(res)
        # As in the paper, we will not trim
        #obj += λ_obj(λ[i, :], moment_vals, weight_i)
        obj += -1 * Optim.minimum(res)
        if grad != nothing
            # Use envelope theorem to obtain gradient
            grad[:] += sum(weight_i[j] * λ[i, :] * moment_jac[j, :, :] / (1 + transpose(λ[i, :]) * moment_vals[j, :]) for j in 1:n_obs, dims=1)[1, :]
        end
    end

    if out != nothing
        return obj
    end
end

function moment_vals_and_jac(n_moments::UInt, moment_func, X::Matrix{T}, Y::Matrix{T}, theta::Vector{T})::Tuple{Matrix{T}, Array{T, 3}} where T <: AbstractFloat
    n_obs = size(X, 1)
    n_params = length(theta)
    moment_vals = Matrix{T}(undef, n_obs, n_moments)
    moment_jac = Array{T, 3}(undef, n_obs, n_moments, n_params)
    out = DiffResults.JacobianResult(zeros(n_moments), theta)
    for i in 1:n_obs
        ForwardDiff.jacobian!(out, x -> moment_func(X[i, :], Y[i, :], x), theta)
        moment_vals[i, :] = DiffResults.value(out)
        moment_jac[i, :, :] = DiffResults.jacobian(out)
    end
    return moment_vals, moment_jac
end

function zero_in_hull_for_all(moments, weights, not_trimmed)
    n_obs = size(moments, 1)
    for i in 1:n_obs
        # For trimmed observations, we don't care as they will drop from the SER
        if not_trimmed[i]
            pos_weight = weights[i, :] .> 0
            if !zero_in_hull(moments[pos_weight, :])
                return false
            end
        end
    end
    return true
end

function zero_in_hull(moments)
    if size(moments, 2) == 1
        return any(moments .>= 0) && any(moments .<= 0)
    end
    # Based on https://stackoverflow.com/a/16898636/7388096
    hull = scipy_spatial.Delaunay(moments)
    return hull.find_simplex(zeros(size(moments, 2))) >= 0
end

function λ_obj_fgh(f, grad, hess, γ, moment_vals, weights)
    n_obs = size(moment_vals, 1)
    if grad != nothing
        grad[:] .= 0
    end
    if hess != nothing
        hess[:, :] .= 0
    end
    out = 0.
    for j in 1:n_obs
        if weights[j] > 0
            inlog = 1 + transpose(γ) * moment_vals[j, :]
            if inlog <= 0
                if grad != nothing
                    grad[:] .= NaN
                end
                if hess != nothing
                    hess[:, :] .= NaN
                end
                if f != nothing
                    return Inf
                end
            end
            if f != nothing
                out -= weights[j] * log(inlog)
            end
            if grad != nothing
                grad[:] -= weights[j] * moment_vals[j, :] / inlog
            end
            if hess != nothing
                hess[:, :] -= weights[j] * moment_vals[j, :] * transpose(moment_vals[j, :]) / inlog
            end
        end
    end
    if f != nothing
        return out
    end
end

end
