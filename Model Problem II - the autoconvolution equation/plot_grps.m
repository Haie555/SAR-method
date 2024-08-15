%--------------------------------------------------------------------------
% Original Code by Author Haie Long
% Modified on [August 14, 2024]
%
% This script performs an iterative scheme to obtain approximate solutions
% for a given problem with noisy data. The solutions are then processed 
% using k-means clustering to identify representative solutions, which 
% are further smoothed and visualized.
%--------------------------------------------------------------------------

clear all; close all

% Grid setup
nel = 257;  % Number of grid points
h = 1/(nel-1);  % Grid spacing
t = [0:h:1]';  % Grid points on the interval [0,1]

% Exact solution computation
xe = zeros(nel, 1);
for i = 1:nel
    xe(i) = cos(2*pi*t(i)) + 30*t(i)*(1-t(i))^3;
end

ye = forward(xe);  % Exact data obtained by applying the forward operator

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Noisy data generation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
delta = 0.05;  
theta = 0.5;
tau = 1.1; 
r = 2;
MaxIter = 100;
noise = randn(nel,1);
yn = ye + delta*noise;  % Add noise to the exact data
deltan = norm(yn - ye);  % L2 norm of the noise
Path = 100;  % Number of sample paths
theta_delta = theta*delta;

% Reconstruction with initial guess x0 = 1*ones(nel,1)
for p = 1:Path  % Loop over all sample paths
    x0 = ones(nel,1);
    xk = x0;
    res = yn - forward(x0);  % Initial residual
    resn = norm(res);  % Norm of the residual
    res_sar(1,p) = resn;
    R = sqrt(mean(res_sar(:,p)));  % Compute initial R
    k = 1;

    while resn > tau*deltan && k < MaxIter
        A = Frechet(xk);  
        res = yn - forward(xk); 
        resn = norm(res);  
        res_r = abs(res).^(r-1).*sign(res);  % Compute the residual in L2 norm
        rhs = A'*res_r;  % Compute the right-hand side for the update
        lam = norm(rhs)^2 / (norm(A*rhs))^2;  % Compute the step size
        dW = sqrt(lam)*randn(nel,1);  % Random perturbation
        xk = xk + lam*rhs + theta_delta*sqrt(1/(1+lam))*R*dW;  % Update the solution
        RE(k,p) = norm(xe-xk)/norm(xe);  % Relative error
        Res(k,p) = resn;  % Store the residual norm
        k = k + 1;
    end
    Xt(:,p) = xk;  % Store the final reconstruction for this path
end
xt_M = mean(Xt,2);  % Mean of reconstructions across all paths

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computation of 90% Confidence Intervals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ConInt = 0.9;
up_xt = zeros(nel,1); 
low_xt = zeros(nel,1);

for k = 1:nel
    mpp = sort(abs(Xt(k,:) - xt_M(k)));  % Sort the deviations
    up_xt(k) = mpp(ceil(ConInt*Path)) + xt_M(k);  % Upper bound of CI
    low_xt(k) = xt_M(k) - mpp(ceil(ConInt*Path));  % Lower bound of CI
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Repeat Reconstruction with Initial Guess x00 = -ones(nel,1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for p = 1:Path  % Loop over all sample paths
    x00 = -ones(nel,1);
    xkk = x00;

    res0 = yn - forward(x00);  % Initial residual
    resn0 = norm(res0);  % Norm of the residual
    res0_sar(1,p) = resn0;
    R = sqrt(mean(res0_sar(:,p)));  % Compute initial R
    k = 1;

    while resn0 > tau*deltan && k < MaxIter
        A = Frechet(xkk);  % Compute the Frechet derivative
        res0 = yn - forward(xkk);  % Update the residual
        resn0 = norm(res0);  % Norm of the updated residual
        res_r = abs(res0).^(r-1).*sign(res0);  % Compute the residual in L2 norm
        rhs = A'*res_r;  % Compute the right-hand side for the update
        lam = norm(rhs)^2 / (norm(A*rhs))^2;  % Compute the step size
        dW = sqrt(lam)*randn(nel,1);  % Random perturbation
        xkk = xkk + lam*rhs + theta_delta*sqrt(1/(1+lam))*R*dW;  % Update the solution
        REE(k,p) = norm(xe-xkk)/norm(xe);  % Relative error
        Ress(k,p) = resn0;  % Store the residual norm
        k = k + 1;
    end
    Xtt(:,p) = xkk;  % Store the final reconstruction for this path
end
xtt_M = mean(Xtt,2);  % Mean of reconstructions across all paths

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computation of 90% Confidence Intervals for x00 Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
up_xtt = zeros(nel,1); 
low_xtt = zeros(nel,1);

for k = 1:nel
    mpp0 = sort(abs(Xtt(k,:) - xtt_M(k)));  % Sort the deviations
    up_xtt(k) = mpp0(ceil(ConInt*Path)) + xtt_M(k);  % Upper bound of CI
    low_xtt(k) = xtt_M(k) - mpp0(ceil(ConInt*Path));  % Lower bound of CI
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting the Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1)
plot(t, xe, 'k', 'LineWidth', 1)  % Plot the true positive solution
hold on
plot(t, -xe, 'b', 'LineWidth', 1.2)  % Plot the true negative solution
hold on
plot(t, low_xt, '--c', 'LineWidth', 1.2)  % Plot lower CI for grp1
hold on
plot(t, up_xt, '--c', 'LineWidth', 1.2)  % Plot upper CI for grp1
hold on
plot(t, low_xtt, '--r', 'LineWidth', 1.2)  % Plot lower CI for grp2
hold on
plot(t, up_xtt, '--r', 'LineWidth', 1.2)  % Plot upper CI for grp2
axis tight;

% Add legend for clarity
legend('True pos xe', 'True neg xe', '90%-CI of grp1', '90%-CI of grp1', '90%-CI of grp2', '90%-CI of grp2');
