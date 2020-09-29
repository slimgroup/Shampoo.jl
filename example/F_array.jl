using PyPlot, JUDI.TimeModeling, LinearAlgebra, FFTW, JOLI, DSP
using IterativeSolvers

using Printf

n = (401, 401)   # (x,y,z) or (x,z)
d = (5f0, 5f0)
o = (0f0, 0f0)

extentx = n[1]*d[1]
extentz = n[2]*d[2]

v = 2f0*ones(Float32,n)

# Slowness squared [s^2/km^2]
m = convert(Array{Float32,2},(1f0 ./ v).^2)	# true model

model = Model(n, d, o, m; nb = 200)

m = reshape(model.m, n[1], n[2], 1, 1)

# Model structure

dtS = 2f0
timeS = 3000f0
nt = Int64(timeS/dtS)+1

wavelet = ricker_wavelet(timeS, dtS, 0.04f0)[:,1]

################################### Source/receiver geometry ######################################

# Set up receiver geometry

domain_x = (model.n[1] - 1)*model.d[1]
domain_z = (model.n[2] - 1)*model.d[2]

nxrec = 512

function circleShape(h, k ,r)
	theta = LinRange(0,2*pi, nxrec)
	h .+ r*sin.(theta), k .+ r*cos.(theta)
end

xrec, zrec = circleShape(domain_x / 2,domain_z/2, domain_z/2-5*d[2])
yrec = 0f0 #2d so always 0

# receiver sampling and recording time
timeR = 3000f0   # receiver recording time [ms]
dtR   = 2f0 
# Set up receiver structure
nsrc = 1
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

# Set up info structure for linear operators
ntComp = get_computational_nt(recGeometry, model)
info = Info(prod(n), nsrc, ntComp)

# Return shots as julia vector (needed for tracker)
opt1 = Options(return_array=true)
opt2 = Options(return_array=false)

Pr  = judiProjection(info, recGeometry)
F1  = judiModeling(info, model; options=opt1)	# modeling operator initial model
F2  = judiModeling(info, model; options=opt2)	# modeling operator initial model
Pw = judiLRWF(info, wavelet)
# Combined operators
F1 = Pr*F1*adjoint(Pw)
F2 = Pr*F2*adjoint(Pw)

q = zeros(Float32,n)
q[201,201] = 1f0
q1 = vec(q)
q2 = judiWeights(q)

d_obs1 = F1*q1
d_obs2 = F2*q2

@printf("d_obs difference %e expect to be 0 - yes \n", (norm(d_obs1-vec(d_obs2.data[1]))/norm(d_obs1)))
@printf("q difference %e expect to be 0 - yes \n", norm(q1-vec(q2.weights[1]))/norm(q1))
@printf("norm d_obs1 %e \n",norm(d_obs1))
@printf("norm d_obs2 %e \n",norm(d_obs2))
@printf("expect them to be the same - no \n")

maxit = 3

x1 = 0f0 .* q1
#x1,his1 = lsqr!(x1,F1,d_obs1,atol=0f0,btol=0f0,conlim=0f0,maxiter=maxit,log=true,verbose=true)
x1,his1 = lsqr(F1,d_obs1,atol=0f0,btol=0f0,conlim=0f0,maxiter=maxit,log=true,verbose=true)

#x2 = 0f0 .* q2
#x2,his2 = lsqr!(x2,F2,d_obs2,atol=0f0,btol=0f0,conlim=0f0,maxiter=maxit,log=true,verbose=true)
x2,his2 = lsqr!(0f0 .* q2,F2,d_obs2,atol=0f0,btol=0f0,conlim=0f0,maxiter=maxit,log=true,verbose=true)
x3,his3 = lsqr!(0f0 .* q2,F2,d_obs2,atol=0f0,btol=0f0,conlim=0f0,maxiter=10,log=true,verbose=true)

@printf("iterative solution difference %e expect to be 0 - no \n",norm(x1-vec(x2.weights[1]))/norm(x1))

figure(); # visualize the difference
subplot(1,2,1);
imshow(reshape(x1,n));
subplot(1,2,2);
imshow(reshape(x2.weights[1],n));

figure();
plot([1;his1[:resnorm]/norm(d_obs1)],label="Array")
plot([1;his2[:resnorm]/norm(d_obs2)],label="judiVector")
legend()

res1 = F1*x1-d_obs1
res2 = F2*x2-d_obs2