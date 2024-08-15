close all; clear;

% Load data
load 2D_data2

% Inversion Parameters
maxit = 200;  
tau = 1.3;  
theta = 0.02;  
Path = 10;  
delta = 0.01;

% Initialize variables
Xt = zeros(nel, Path);
IterNum = zeros(Path, 1);

for p = 1:Path
    % Initial guess
    x0 = ones(nel, 1); 
    [S, My, rhs, rrhs, dy, res_y] = forward_2D(fem, x0, Mf, u_delta);
    residual = norm(res_y);
    
    % Store initial error metrics
    RE_SAR(1, p) = norm(x0 - x_true) / norm(x_true);
    Error_SAR(1, p) = norm(x0 - x_true);
    Residual_SAR(1, p) = residual;
    
    xt = x0;
    k = 1;
    R = sqrt(mean(Residual_SAR(:, p)));
    dt = norm(rhs)^2 / norm(dy)^2;
    
    % Iterative inversion process
    while residual > tau * deltan && k < maxit
        dW = sqrt(dt) * randn(nel, 1);  % Wiener process increment
        [S, My, rhs, rrhs, dy, res_y] = forward_2D(fem, xt, Mf, u_delta);
        dt = norm(rhs)^2 / norm(dy)^2;
        
        % Update solution with stochastic perturbation
        xt = xt + dt * rhs + theta * sqrt(1 / (1 + dt)) * R * dW;
        
        % Store error metrics
        RE_SAR(k + 1, p) = norm(xt - x_true) / norm(x_true);
        Error_SAR(k + 1, p) = norm(xt - x_true);
        residual = norm(res_y);
        Residual_SAR(k + 1, p) = residual;
        
        k = k + 1;
    end
    
    IterNum(p) = k;  % Store number of iterations
    Xt(:, p) = xt;   % Store final reconstruction
end

% Best-Path Reconstruction by Minimum Norm Solution
BPR1 = zeros(Path, 1);
for p = 1:Path
    BPR1(p) = norm(Xt(:, p) - x_true);
end

[MinNorm, P1] = min(BPR1);  % Find path with minimum norm
BPR_MinNorm = Xt(:, P1);    % Best path reconstruction
RE_MinNorm = RE_SAR(:, P1); % Corresponding relative error

% Compute reconstruction statistics
xt_M = mean(Xt, 2);
MeanNorm = norm(xt_M - x_true);

% Compute average metrics
Residual_M_SAR = mean(Residual_SAR, 2);
RE_M_SAR = mean(RE_SAR, 2);
Error_M_SAR = mean(Error_SAR, 2);

% Plot residual and error over iterations
figure(1)
plot(Residual_M_SAR, 'r'); axis tight; title('Residual');

figure(2)
plot(Error_M_SAR, 'r'); axis tight; title('Error');

figure(2)
plot(RE_M_SAR, 'r'); axis tight; title('Relative error');

% Show reconstruction
figure(4)
fem.plotU(xt_M);
shading interp; camlight headlight; lighting phong; axis tight; drawnow;


