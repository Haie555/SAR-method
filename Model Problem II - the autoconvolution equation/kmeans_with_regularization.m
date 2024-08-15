% Functions
function [idx, C] = kmeans_with_regularization(X, k, lambda, opts)
    % X: data points (rows are data points)
    % k: number of clusters
    % lambda: regularization parameter
    % opts: options for k-means

    % Initialize cluster centers randomly
    [num_points, num_features] = size(X);
    rand_indices = randperm(num_points, k);
    C = X(rand_indices, :);

    for iter = 1:opts.MaxIter
        % Step 2: Assign each point to the nearest cluster center with regularization
        idx = zeros(num_points, 1);
        for i = 1:num_points
            min_dist = Inf;
            for j = 1:k
                % Compute distance with regularization term
%                 dist = norm(X(i, :) - C(j, :))^2 + lambda * norm(diff(X(i, :)) - diff(C(j, :)))^2;
                dist = norm(X(i, :) - C(j, :))^2 + lambda * norm(diff(X(i, :)))^2;
                if dist < min_dist
                    min_dist = dist;
                    idx(i) = j;
                end
            end
        end

        % Step 3: Update cluster centers
        new_C = zeros(k, num_features);
        for j = 1:k
            cluster_points = X(idx == j, :);
            if ~isempty(cluster_points)
                new_C(j, :) = mean(cluster_points, 1);
            else
                new_C(j, :) = C(j, :);
            end
        end

        % Check for convergence
        if norm(new_C - C) < 1e-5
            break;
        end

        C = new_C;
    end
end
