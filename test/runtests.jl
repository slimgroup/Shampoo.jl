using Shampoo
using JUDI, Test
using LinearAlgebra, Printf, JOLI

include("test_utils.jl")

basic = ["test_adjoint.jl","test_linearity.jl", "test_output.jl"]

@testset "Shampoo.jl" begin
    for t = basic
        include(t)
    end
end