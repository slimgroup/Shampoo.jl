using SeismicPreconditioners
using JUDI.TimeModeling, Test
using LinearAlgebra, Printf, JOLI

include("test_utils.jl")

basic = ["test_adjoint.jl"]

@testset "SeismicPreconditioners.jl" begin
    for t = basic
        include(t)
    end
end