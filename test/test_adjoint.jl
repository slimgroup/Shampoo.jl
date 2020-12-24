# Adjoint test for preconditioners
# Author: Ziyi Yin, ziyi.yin@gatech.edu
# Date: Dec 2020
#

using JUDI.TimeModeling, LinearAlgebra, Test, Distributed, Printf

### Model
model, model0, dm = setup_model()
nsrc = 4
q, srcGeometry, recGeometry, info = setup_geom(model; nsrc=nsrc)
dt = srcGeometry.dt[1]

tol = 5f-4
nrec = length(recGeometry.xloc[1])
nt = recGeometry.nt[1]

######## Test left preconditioners

C = CumsumOp(nt,nsrc,nrec)
D = DiffOp(nt,nsrc,nrec)
FInt = FractionalIntegrationOp(nt,nsrc,nrec,0.8)

left_list = [C,D,FInt,C',D',FInt]

data1 = Array{Array}(undef, nsrc)
data2 = Array{Array}(undef, nsrc)
for j=1:nsrc
    data1[j] = randn(Float32, nt, nrec)
    data2[j] = randn(Float32, nt, nrec)
end
d_obs1 = judiVector(recGeometry,data1)
d_obs2 = judiVector(recGeometry,data2)

for Op in left_list
    a = dot(Op*d_obs1, d_obs2)
    b = dot(d_obs1,Op'*d_obs2)
    @printf(" <F x, y> : %2.5e, <x, F' y> : %2.5e, relative error : %2.5e \n", a, b, (a - b)/(a + b))
    @test isapprox(a/(a+b), b/(a+b), atol=tol, rtol=0)
end

######## Test right preconditioners

M = MaskOp(model.n,randn(Float32,model.n))

right_list = [M, M']

dm1 = randn(Float32,prod(model.n))
dm2 = randn(Float32,prod(model.n))

for Op in right_list
    a = dot(Op*dm1, dm2)
    b = dot(dm1,Op'*dm2)
    @printf(" <F x, y> : %2.5e, <x, F' y> : %2.5e, relative error : %2.5e \n", a, b, (a - b)/(a + b))
    @test isapprox(a/(a+b), b/(a+b), atol=tol, rtol=0)
end