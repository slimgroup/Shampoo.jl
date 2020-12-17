using PyPlot, JUDI.TimeModeling, LinearAlgebra, FFTW, JOLI, DSP
using IterativeSolvers

using SeismicPreconditioners

n = (401, 401)   # (x,y,z) or (x,z)
d = (5f0, 5f0)
o = (0f0, 0f0)

extentx = n[1]*d[1]
extentz = n[2]*d[2]

v = 2f0*ones(Float32,n)

# Slowness squared [s^2/km^2]
m = convert(Array{Float32,2},(1f0 ./ v).^2)	# true model

model = Model(n, d, o, m; nb = 200)

# Model structure

dtS = 1f0
timeS = 3000f0
nt = Int64(timeS/dtS)+1

fmin = 10f0
fmax = 60f0
cfreqs = (fmin, fmin+10f0, fmax-10f0, fmax) # corner frequencies
wavelet = reshape(cfreq_wavelet(500, nt, dtS/1f3, cfreqs; edge=hamming),nt,1)
#wavelet = ricker_wavelet(timeS, dtS, 0.04f0)[:,1]

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
dtR   = 1f0 
# Set up receiver structure
nsrc = 1
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

mute_matrix = zeros(Float32,n)

for i = 1:n[1]
	for j = 1:n[2]
		mute_matrix[i,j] = ((i-201)^2+(j-201)^2<190^2)
	end
end

M = MaskOp(n,mute_matrix)

# Set up info structure for linear operators
ntComp = get_computational_nt(recGeometry, model)
info = Info(prod(n), nsrc, ntComp)

# Return shots as julia vector (needed for tracker)
opt = Options(free_surface=false,isic=false)

Pr  = judiProjection(info, recGeometry)
F  = judiModeling(info, model; options=opt)	# modeling operator initial model
Pw = judiLRWF(info, wavelet)
# Combined operators
F = Pr*F*adjoint(Pw)

#P = FractionalIntegrationOp(nt,nsrc,nxrec,0.95)
P = DiffOp(nt,nsrc,nxrec)

# Extended modeling CNN layers
q = zeros(Float32,n)
q[201,201] = 1f0
q = judiWeights(q)

PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=8); PyPlot.rc("ytick", labelsize=8)
figure(); imshow(reshape(sqrt.(1f0./m), n[1], n[2])',extent=(0,extentx,extentz,0));
PyPlot.scatter(xrec,zrec,marker=".",color="red",label="receivers",s=1);
PyPlot.scatter(201*d[1],201*d[2],label="sources",marker="x",color="white");
legend(loc=3,fontsize=8);
xlabel("Lateral position [m]", fontsize=8);ylabel("Horizontal position [m]", fontsize=8);
fig = PyPlot.gcf();
title("Experimental setup")

######################################### Set up JUDI operators ###################################

d_obs = F*q
q_new = F'*d_obs

q_precond = F'*P'*P*d_obs

figure();
subplot(2,1,1);
PyPlot.plot(wavelet[1:1000]);title("zoomed-in wavelet in time domain");
subplot(2,1,2);
PyPlot.plot(abs.(fftshift(fft(wavelet))));title("wavelet in frequency domain");

figure();
subplot(2,2,1)
imshow(reshape(q_new.weights[1],n)[130:270,130:270]',vmin=-0.8*norm(q_new.weights[1],Inf),vmax=0.8*norm(q_new.weights[1],Inf));title("F'*F*q")
subplot(2,2,2)
imshow(reshape(q_precond.weights[1],n)[130:270,130:270]',vmin=-0.8*norm(q_precond.weights[1],Inf),vmax=0.8*norm(q_precond.weights[1],Inf));title("F'*P'*P*F*q")
subplot(2,2,3)
imshow(abs.(fftshift(fft(reshape(q_new.weights[1],n)')))/norm(abs.(fftshift(fft(reshape(q_new.weights[1],n)'))),Inf),cmap="jet",vmin=0,vmax=1)
subplot(2,2,4)
imshow(abs.(fftshift(fft(reshape(q_precond.weights[1],n)')))/norm(abs.(fftshift(fft(reshape(q_precond.weights[1],n)'))),Inf),cmap="jet",vmin=0,vmax=1)

maxit = 10

q1 = 0f0 .* q
q1,his1 = lsqr!(q1,F*M,d_obs,atol=0f0,btol=0f0,conlim=0f0,maxiter=maxit,log=true,verbose=true)
q2 = 0f0 .* q
q2,his2 = lsqr!(q2,P*F*M,P*d_obs,atol=0f0,btol=0f0,conlim=0f0,maxiter=maxit,log=true,verbose=true)
res1 = his1[:resnorm]
res2 = his2[:resnorm]

u1 = 20*log.(res1./norm(d_obs))
u2 = 20*log.(res2./norm(P*d_obs))

figure();
xlabel("iterations")
ylabel("ratio")
PyPlot.plot(res1/norm(d_obs),label="LSQR")
PyPlot.plot(res2/norm(P*d_obs),label="P-LSQR")
title("normalized least squares residual")
legend()


figure();
xlabel("iterations")
ylabel("dB")
PyPlot.plot(u1,label="LSQR")
PyPlot.plot(u2,label="P-LSQR")
title("normalized log-based least squares residual")
legend()


figure();
subplot(2,2,1)
imshow(reshape((M*q1).weights[1],n)[130:270,130:270]',vmin=-0.8*norm((M*q1).weights[1],Inf),vmax=0.8*norm((M*q1).weights[1],Inf));title("LSQR")
subplot(2,2,2)
imshow(reshape((M*q2).weights[1],n)[130:270,130:270]',vmin=-0.8*norm((M*q2).weights[1],Inf),vmax=0.8*norm((M*q2).weights[1],Inf));title("P-LSQR")
subplot(2,2,3)
imshow(abs.(fftshift(fft(reshape((M*q1).weights[1],n)'))),cmap="jet")
subplot(2,2,4)
imshow(abs.(fftshift(fft(reshape((M*q2).weights[1],n)'))),cmap="jet")
