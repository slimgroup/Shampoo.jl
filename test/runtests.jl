using SeismicPreconditioners
using JUDI.TimeModeling, Test
using LinearAlgebra, Printf, JOLI

include("test_utils.jl")

basic = ["test_adjoint.jl","test_linearity.jl", "test_output.jl"]

@testset "SeismicPreconditioners.jl" begin
    for t = basic
        include(t)
    end
end