using PyPlot, JUDI.TimeModeling, LinearAlgebra, FFTW, JOLI, DSP
using IterativeSolvers

using SeismicPreconditioners

function weight(n, d, xsrc_index, zsrc_index, nsrc)
	xs = reshape(xsrc_index*d[1],:,1,nsrc);
	zs = reshape(zsrc_index*d[1],1,:,nsrc);
	x = zeros(n[1],1,nsrc);
	x .= reshape((collect(1:n[1]).-1)*d[1], :, 1)
	z = zeros(1,n[2],nsrc);
    z .= reshape((collect(1:n[2]).-1)*d[2], 1, :)
	W = zeros(Float32,n[1],n[2],nsrc)
	#W .= sqrt.((x.-xs).^2f0)
	W .= sqrt.((x.-xs).^2f0.+(z.-zs).^2f0)
	W = vec(W)
	N = prod(n)
	P = joLinearFunctionFwd_T(N, N,
	                             v -> W.*v,
	                             w -> W.*w,
	                             Float32,Float32,name="Horizontal Penalty")
	return P
end

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

dtS = 0.5f0
timeS = 3000f0
nt = Int64(timeS/dtS)+1

fmin = 10f0
fmax = 80f0
cfreqs = (fmin, fmin+10f0, fmax-10f0, fmax) # corner frequencies
wavelet = cfreq_wavelet(500, nt, dtS/1f3, cfreqs; edge=hamming)
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

#xrec = range(d[1], stop=Float32((n[1]-1)*d[1]), length=nxrec)
#zrec = range(0f0, stop=0f0, length=nxrec)

# receiver sampling and recording time
timeR = 3000f0   # receiver recording time [ms]
dtR   = 0.5f0 
# Set up receiver structure
nsrc = 1
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

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

P = RightPrecondF(F)
######################################### Neural network ##########################################
# Extended modeling CNN layers
q = zeros(Float32,n)
q[201,201] = 1f0
q = judiWeights(q)

PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=8); PyPlot.rc("ytick", labelsize=8)
figure(); imshow(reshape(sqrt.(1f0./m), n[1], n[2])',extent=(0,extentx,extentz,0));
#cb = colorbar(fraction=0.027, pad=0.04,format="%.4f");
#cb = colorbar(fraction=0.027, pad=0.04);
#cb[:set_label]("Velocity [km/s]", labelsize=20)
#cb[:set_label]("Squared slowness [km^2/s^2]")
PyPlot.scatter(xrec,zrec,marker=".",color="red",label="receivers",s=1);
PyPlot.scatter(201*d[1],201*d[2],label="sources",marker="x",color="white");
legend(loc=3,fontsize=8);
xlabel("Lateral position [m]", fontsize=8);ylabel("Horizontal position [m]", fontsize=8);
fig = PyPlot.gcf();
title("Experimental setup")

######################################### Set up JUDI operators ###################################

q_new = F'*F*q

q_precond = P'*F'*F*P*q

figure();
subplot(2,1,1);
PyPlot.plot(wavelet[1:1000]);title("zoomed-in wavelet in time domain");
subplot(2,1,2);
PyPlot.plot(abs.(fftshift(fft(wavelet))));title("wavelet in frequency domain");

figure();
subplot(2,2,1)
imshow(reshape(q_new.data[1],n)[130:270,130:270]',vmin=-0.8*norm(q_new.data[1],Inf),vmax=0.8*norm(q_new.data[1],Inf));title("F'*F*q")
subplot(2,2,2)
imshow(reshape(q_precond.data[1],n)[130:270,130:270]',vmin=-0.8*norm(q_precond.data[1],Inf),vmax=0.8*norm(q_precond.data[1],Inf));title("P'*F'*F*P*q")
subplot(2,2,3)
imshow(abs.(fftshift(fft(reshape(q_new.data[1],n)')))/norm(abs.(fftshift(fft(reshape(q_new.data[1],n)'))),Inf),cmap="jet",vmin=0,vmax=1)
subplot(2,2,4)
imshow(abs.(fftshift(fft(reshape(q_precond.data[1],n)')))/norm(abs.(fftshift(fft(reshape(q_precond.data[1],n)'))),Inf),cmap="jet",vmin=0,vmax=1)

d_obs = F*q

maxit = 20

q1,his1 = lsqr(F,d_obs,atol=0f0,btol=0f0,conlim=0f0,maxiter=maxit,log=true,verbose=true)

q2,his2 = lsqr(F*L,d_obs,atol=0f0,btol=0f0,conlim=0f0,maxiter=maxit,log=true,verbose=true)

res1 = his1[:resnorm]
res2 = his2[:resnorm]

u1 = 20*log.(res1./norm(d_obs))
u2 = 20*log.(res2./norm(d_obs))
u3 = 20*log.(res3./norm(P*d_obs))

figure();
xlabel("iterations")
ylabel("ratio")
PyPlot.plot(res1/norm(d_obs),label="LSQR")
PyPlot.plot(res2/norm(d_obs),label="P-LSQR")
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
imshow(reshape(q1.data[1],n)[130:270,130:270]',vmin=-0.8*norm(q1.data[1],Inf),vmax=0.8*norm(q1.data[1],Inf));title("LSQR")
subplot(2,2,2)
imshow(reshape(L*q2.data[1],n)[130:270,130:270]',vmin=-0.8*norm(L*q2.data[1],Inf),vmax=0.8*norm(L*q2.data[1],Inf));title("P-LSQR")
subplot(2,2,3)
imshow(abs.(fftshift(fft(reshape(q1.data[1],n)'))),cmap="jet")
subplot(2,2,4)
imshow(abs.(fftshift(fft(reshape(L*q2.data[1],n)'))),cmap="jet")