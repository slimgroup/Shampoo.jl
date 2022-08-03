using Shampoo
using Test
using LinearAlgebra, JOLI

nt = 21
dt = 1f0
nsrc = 2
nrec = 4

@testset "Shampoo.jl" begin

    P = FractionalIntegrationOp(nt,dt,nsrc,nrec)
    @test isadjoint(P)[1]
    @test islinear(P)[1]

end