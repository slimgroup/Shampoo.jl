# JUDI precon make sure it runs
model, model0, dm = setup_model()
q, src_geometry, rec_geometry, info = setup_geom(model)

ftol = 5f-5

nsrc = q.nsrc
w = judiWeights(randn(Float32, model0.n))
w_out = similar(w)

data = Array{Array}(undef, nsrc)
for j=1:nsrc
    data[j] = randn(Float32, rec_geometry.nt[j], length(rec_geometry.xloc[j]))
end

dobs = judiVector(rec_geometry, data)
dobs_out = similar(dobs)

F = judiFilter(rec_geometry, .002, .030)
Md = judiMarineTopmute2D(0, rec_geometry)
D = judiDepthScaling(model)
M = judiTopmute(model.n, 20, 1)

mul!(dobs_out, F, dobs)
@test isapprox(dobs_out, F*dobs; rtol=ftol)
mul!(dobs_out, F', dobs)
@test isapprox(dobs_out, F'*dobs; rtol=ftol)
mul!(dobs_out, Md, dobs)
@test isapprox(dobs_out, Md*dobs; rtol=ftol)
mul!(dobs_out, Md', dobs)
@test isapprox(dobs_out, Md'*dobs; rtol=ftol)


mul!(w_out, D, w)
@test isapprox(w_out, D*w; rtol=ftol)
@test isapprox(w_out.weights[1][end, end]/w.weights[1][end, end], sqrt((model.n[2]-1)*model.d[2]); rtol=ftol)
mul!(w_out, D', w)
@test isapprox(w_out, D'*w; rtol=ftol)
@test isapprox(w_out.weights[1][end, end]/w.weights[1][end, end], sqrt((model.n[2]-1)*model.d[2]); rtol=ftol)

mul!(w_out, M, w)
@test isapprox(w_out, M*w; rtol=ftol)
@test isapprox(norm(w_out.weights[1][:, 1:19]), 0f0; rtol=ftol)
mul!(w_out, M', w)
@test isapprox(w_out, M'*w; rtol=ftol)

# test the output of depth scaling and topmute operators, and test if they are out-of-place
        
dobs1 = deepcopy(dobs)
for Op in [F, F', Md , Md']
    m = Op*dobs 
    # Test that dobs wasn't modified
    @test isapprox(dobs,dobs1,rtol=eps())
    # Test that it did compute something 
    @test m != dobs
end

w1 = deepcopy(w)

for Op in [D, D']
    m = Op*w
    # Test that w wasn't modified
    @test isapprox(w,w1,rtol=eps())

    w_expect = deepcopy(w)
    for j = 1:model.n[2]
        w_expect.weights[1][:,j] = w.weights[1][:,j] * Float32(sqrt(model.d[2]*(j-1)))
    end
    @test isapprox(w_expect,m)
end

for Op in [M , M']
    m = Op*w
    # Test that w wasn't modified
    @test isapprox(w,w1,rtol=eps())

    @test all(isapprox.(m.weights[1][:,1:18], 0))
    @test isapprox(m.weights[1][:,21:end],w.weights[1][:,21:end])
end

        
n = (100,100,100)
d = (10f0,10f0,10f0)
o = (0.,0.,0.)
m = 0.25*ones(Float32,n)
model3D = Model(n,d,o,m)

D3 = judiDepthScaling(model3D)

dm = randn(Float32,prod(n))
dm1 = deepcopy(dm)
for Op in [D3, D3']
    opt_out = Op*dm
    # Test that dm wasn't modified
    @test dm1 == dm

    dm_expect = zeros(Float32,model3D.n)
    for j = 1:model3D.n[3]
        dm_expect[:,:,j] = reshape(dm,model3D.n)[:,:,j] * Float32(sqrt(model3D.d[3]*(j-1)))
    end
    @test isapprox(vec(dm_expect),opt_out)
end

M3 = judiTopmute(model3D.n, 20, 1)

for Op in [M3, M3']
    opt_out = Op*dm
    # Test that dm wasn't modified
    @test dm1 == dm

    @test all(isapprox.(reshape(opt_out,model3D.n)[:,:,1:18], 0))
    @test isapprox(reshape(opt_out,model3D.n)[:,:,21:end],reshape(dm,model3D.n)[:,:,21:end])
end

# test find_water_bottom in 3D

dm3D = zeros(Float32,10,10,10)
dm3D[:,:,4:end] .= 1f0
@test find_water_bottom(dm3D) == 4*ones(Integer,10,10)
