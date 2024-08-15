%--------------------------------------------------------------------------
% Revised on 2024.08.13
%--------------------------------------------------------------------------

clear; clc;
load data001

% Inversion Parameters
maxit = 200;         % Maximum iterations
theta = 0.1;         
Path = 20;           % Number of sample paths
tau = 1.008;         

% Initialize variables
Xt = zeros(nel, Path);
Res = zeros(Path, 1);

for p = 1:Path
    % Initial guess
    x0 = fem.P(2 + 0 * x);
    [S, My, rhs, rrhs, dy, res_y] = forward_1D(fem, x0, Mf, u_delta);
    residual = norm(res_y);
    
    % Store initial error metrics
    RE_sar(1, p) = norm(x0 - x_true) / norm(x_true);
    error_sar(1, p) = norm(x0 - x_true);
    residual_sar(1, p) = residual;
    
    xt = x0;
    k = 1;
    R = sqrt(mean(residual_sar(:, p)));
    dt = norm(rhs)^2 / norm(dy)^2;
    
    % Iterative inversion process
    while residual > tau * deltau && k < maxit
        dW = sqrt(dt) * randn(nel, 1);  % Wiener process increment
        [S, My, rhs, rrhs, dy, res_y] = forward_1D(fem, xt, Mf, u_delta);
        dt = norm(rhs)^2 / norm(dy)^2;
        
        % Update solution with stochastic perturbation
        xt = xt + dt * rhs + theta * sqrt(1 / (1 + dt)) * R * dW;
        
        % Store error metrics
        RE_sar(k, p) = norm(xt - x_true) / norm(x_true);
        error_sar(k, p) = norm(xt - x_true);
        residual = norm(res_y);
        residual_sar(k, p) = residual;
        
        k = k + 1;
    end
    
    Xt(:, p) = xt;    % Store final reconstruction
    Res(p) = norm(xt - x_true); % Store final error
end

% Compute mean and confidence intervals for the reconstruction
xt_M = mean(Xt, 2);
up_xt = zeros(nel, 1);
low_xt = zeros(nel, 1);

for k = 1:nel
    MP_k = Xt(k, :) - xt_M(k);
    sort_MP = sort(MP_k);
    
    % Compute confidence intervals
    up_xt(k) = sort_MP(ceil(0.9 * Path)) + xt_M(k);
    low_xt(k) = sort_MP(floor(0.1 * Path)) + xt_M(k);
end

% Compute mean residual, relative error, and error
NaRes = residual_sar ~= 0;
NbRes = sum(NaRes, 2);
Residual_M_SAR = sum(residual_sar, 2) ./ NbRes;

NaRE = RE_sar ~= 0;
NbRE = sum(NaRE, 2);
RE_M_SAR = sum(RE_sar, 2) ./ NbRE;

NaErr = error_sar ~= 0;
NbErr = sum(NaErr, 2);
Error_M_SAR = sum(error_sar, 2) ./ NbErr;

% Plot results
figure(1)
plot(Residual_M_SAR, 'r'); title('Residual');

figure(2)
plot(Error_M_SAR, 'r'); title('Error');

figure(6)
plot(xc, x_true, 'k', xc, xt_M, '--m', xc, low_xt, '--b', xc, up_xt, '--r');
axis tight;
legend('Exact solution', 'Expectation of SAR', ...
    'Lower bound of 80% confidence interval', ...
    'Upper bound of 80% confidence interval');
