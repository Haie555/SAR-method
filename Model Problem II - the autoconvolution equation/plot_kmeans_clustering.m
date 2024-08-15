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

% Parameters
delta = 0.05;  % Noise level
theta = 0.5;  % Regularization parameter for noise
theta_delta = delta * theta;
r = 2;  % L2 norm
MaxIter = 100;  % Maximum number of iterations
Path = 100;  % Number of sample paths
tau = 1.1;  % Threshold parameter
lambda = 0.1;  % Regularization parameter for k-means

% Compute the exact solution xe
xe = zeros(nel, 1);
for i = 1:nel
    xe(i) = cos(2*pi*t(i)) + 30*t(i)*(1-t(i))^3;
end

% Compute the exact data y(t)
ye = forward(xe);

% Generate noisy data
noise = randn(nel, 1);
yn = ye + delta * noise;  % Add noise to the exact data
deltan = norm(yn - ye);  % L2 norm of the noise

% Initialize storage for valid solutions
approx_solutions = zeros(nel, Path);
valid_solutions_count = 0;

% Iterative scheme to obtain approximate solutions
for p = 1:Path
    % Choose an initial guess
    x0 = -ones(nel, 1);  % Uniform initial guess
    xk = x0;
    res = yn - forward(xk);  % Compute residual
    resn = norm(res);
    res_sar(1,p) = resn;
    R = sqrt(mean(res_sar(:,p)));
    iter = 1;
    
    % Iterative process
    while resn > tau * deltan && iter < MaxIter
        A = Frechet(xk);  % Compute Frechet derivative
        res = yn - forward(xk);  % Update residual
        resn = norm(res);  % Update norm of residual
        res_r = abs(res).^(r-1) .* sign(res);  % Compute residual in L2 norm
        rhs = A' * res_r;  % Compute right-hand side for the update
        lam = norm(rhs)^2 / (norm(A * rhs))^2;  % Compute step size
        dW = sqrt(lam) * randn(nel,1);  % Random perturbation
        xk = xk + lam * rhs + theta_delta * sqrt(1/(1+lam)) * R * dW;  % Update solution
        RE(iter, p) = norm(xe - xk) / norm(xe);  % Compute relative error
        Res(iter, p) = resn;  % Store residual norm
        iter = iter + 1;
    end
    
    % Save only satisfactory solutions
    if resn <= tau * deltan
        valid_solutions_count = valid_solutions_count + 1;
        approx_solutions(:, valid_solutions_count) = xk;
    end
end

% Trim the storage matrix to keep only valid solutions
approx_solutions = approx_solutions(:, 1:valid_solutions_count);

% Check if there are enough valid solutions for clustering
if valid_solutions_count > 1
    % Clustering approximate solutions using k-means with regularization
    num_clusters = min(2, valid_solutions_count);  % Ensure num_clusters <= valid_solutions_count
    norm_approx_solutions = normalize(approx_solutions, 1);  % Normalize solutions for clustering
    opts = statset('MaxIter', 500);  % Set options for k-means
    [idx, C] = kmeans(norm_approx_solutions', num_clusters, 'Distance', 'sqeuclidean', 'Replicates', 10, 'Options', opts);
    
    % Select representatives of the clusters
    representatives = zeros(nel, num_clusters);
    for i = 1:num_clusters
        cluster_points = norm_approx_solutions(:, idx == i);  % Get points in the cluster
        center = C(i, :)';  % Cluster center
        distances = vecnorm(cluster_points - center);  % Calculate distances to the center
        [~, min_idx] = min(distances);  % Find the point closest to the center
        representative_idx = find(idx == i, min_idx, 'first');  % Get the index of the representative point
        representatives(:, i) = approx_solutions(:, representative_idx(1));  % Store representative
    end
    
    % Smooth the representative solutions using a moving average filter
    window_size = 20;  % Set window size for smoothing
    smoothed_representatives = zeros(size(representatives));
    for i = 1:num_clusters
        smoothed_representatives(:, i) = smooth(representatives(:, i), window_size);  % Apply smoothing
    end
    
    % Plot the true solution and the smoothed representatives
    figure(1)
    plot(t, xe, 'k', 'LineWidth', 1.2);
    title('True solution');
    
    figure(2)
    plot(t, smoothed_representatives(:, 1), 'r', 'LineWidth', 1.2);
    title('Smoothed representative of cluster');
    
else
    disp('Not enough valid solutions for clustering.');
end

% Randomly select one solution from the valid set
if valid_solutions_count > 0
    random_index = randi(valid_solutions_count);  % Randomly select an index
    random_solution1 = approx_solutions(:, random_index);  % Get the corresponding solution
    
    figure(3)
    plot(t, random_solution1, 'b', 'LineWidth', 1.2);
    title('Randomly selected solution');
else
    disp('No valid solutions found.');
end

