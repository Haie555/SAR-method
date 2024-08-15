function [RE_Land, Res_Land, Error_Land] = Func_Land_delta3(rand_x0, maxit, tau)

% Load the data for delta = 0.01
load data005 

% Initialize variables
[~, ~, ~, ~, ~, res_y] = forward_1D(fem, rand_x0, Mf, u_delta);  % Compute the initial forward operation
Residual = norm(res_y);  % Initial residual norm
xt = rand_x0;  % Set initial guess as the input random initial guess

k = 1;  % Initialize iteration counter
while Residual > tau * deltau && k < maxit
    
    % Store metrics for the current iteration
    RE_Land(k) = norm(xt - x_true) / norm(x_true);  % Relative error
    Error_Land(k) = norm(xt - x_true);  % Absolute error
    Residual = norm(res_y);  % Update residual norm
    Res_Land(k) = Residual;  % Store residual
    
    % Perform forward operation and update the solution
    [~, ~, rhs, ~, dy, res_y] = forward_1D(fem, xt, Mf, u_delta);  % Compute forward operation at current iteration
    dt = norm(rhs)^2 / (norm(dy))^2;  % Adaptive step size
    xt = xt + dt * rhs;  % Update the solution using the Landweber iteration rule
    
    % Increment iteration counter
    k = k + 1;
end

% Convert output metrics to column vectors
Error_Land = Error_Land';
RE_Land = RE_Land';
Res_Land = Res_Land';

end
