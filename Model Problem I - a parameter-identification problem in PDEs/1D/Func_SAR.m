function [Error_SAR, BPR1, RE_BP_SAR, Res_BP_SAR, IterNum, MeanNorm, MinNorm] = Func_SAR(rand_x0, maxit, theta, Path, tau, M)

% Load the data for delta = 0.02
load data002  

% Initialize variables
Xt = zeros(nel, Path);  % Storage for the solutions across paths
IterNum = zeros(Path, 1);  % Storage for the number of iterations per path
given_x0 = fem.P(1 + 0 * x);  % Initial guess provided by the data
[~, ~, ~, ~, ~, res_y] = forward_1D(fem, given_x0, Mf, u_delta);
R0 = norm(res_y);  % Initial residual norm

% Perform SAR iterations across all paths
for p = 1:Path
    x0 = rand_x0;  % Initialize with random starting point
    for k = 1:M
        % Perform M iterations of the initial guess adjustment
        [~, ~, rhs, ~, ~, res_y] = forward_1D(fem, x0, Mf, u_delta);
        Residual = norm(res_y);
        dt = 0.01;  % Step size for iteration
        z0 = given_x0 - x0 - dt * rhs;
        t1 = randn(nel, 1);
        dW = z0 .* abs(t1);    
        x0 = x0 + dt * rhs + dW;  % Update x0 with the step
        RE_SAR(k, p) = norm(x0 - x_true) / norm(x_true);  % Compute relative error
        Error_SAR(k, p) = norm(x0 - x_true);  % Compute error
        Res_SAR(k, p) = Residual;  % Store residual
    end
    
    % Start the main SAR iteration process after M steps
    xt = x0;
    [~, ~, ~, ~, ~, res_y] = forward_1D(fem, xt, Mf, u_delta);
    Residual = norm(res_y);
    
    k = M;  % Start counting iterations from M+1
    while Residual > tau * deltau && k < maxit
        t2 = randn(nel, 1);
        dW = sqrt(dt) * abs(t2);  % Random perturbation
        [S, My, rhs, rrhs, dy, res_y] = forward_1D(fem, xt, Mf, u_delta);
        dt = norm(rhs)^2 / (norm(dy))^2;  % Adaptive step size
        R = min(norm(res_y), R0);  % Update R with the minimum of the current and initial residual norm
        
        % Main update step for SAR
        xt = xt + dt * rhs + delta * theta * sqrt(1 / (1 + dt)) * R * dW;
       
        RE_SAR(k, p) = norm(xt - x_true) / norm(x_true);  % Compute relative error
        Error_SAR(k, p) = norm(xt - x_true);  % Compute error
        Residual = norm(res_y);  % Update residual norm
        Res_SAR(k, p) = Residual;  % Store residual norm
        k = k + 1;
    end
    Xt(:, p) = xt;  % Store final solution for the path
    IterNum(p) = k;  % Store the number of iterations taken
end
xt_M = mean(Xt, 2);  % Compute the mean of the solutions across all paths

% Best-Path Reconstruction by Minimum Norm Solution
BPR1 = zeros(Path, 1);
for p = 1:Path
    BPR1(p) = norm(Xt(:, p) - x_true);  % Compute the norm of the error for each path
end
[MinNorm, P1] = min(BPR1);  % Find the minimum norm solution
BPR_MinNorm = Xt(:, P1);  % The solution with the minimum norm
RE_MinNorm = RE_SAR(:, P1);  % The relative error of the minimum norm solution
MeanNorm = mean(BPR1);  % Mean of the norms across all paths

% Best-Path Reconstruction by Minimum Residual Solution
BPR2 = zeros(Path, 1);
for p = 1:Path
    [~, ~, ~, ~, ~, res_y] = forward_1D(fem, Xt(:, p), Mf, u_delta);
    BPR2(p) = norm(res_y);  % Compute the norm of the residual for each path
end
[MinRes, P2] = min(BPR2);  % Find the minimum residual solution
BPR_MinRes = Xt(:, P2);  % The solution with the minimum residual

% Expectation of Reconstruction
xt_M = mean(Xt, 2);  % Compute the mean solution across all paths
[ZZ, MZ] = min(RE_SAR);  % Find the minimum relative error path
[BPValue, BestPath] = min(MZ);  % Best path value and index

% Extract the best path's relative error and residual sequences
t1 = RE_SAR(:, BestPath);
t2 = find(t1 ~= 0);
t3 = length(t2);
RE_BP_SAR = t1(1:t3);  % Relative error for the best path

r1 = Res_SAR(:, BestPath);
r2 = find(r1 ~= 0);
r3 = length(r2);
Res_BP_SAR = r1(1:r3);  % Residual for the best path

end
