%--------------------------------------------------------------------------
% Revised on 2024.08.13
%
% This script generates probability plots for the event where the 
% approximation error exceeds a given noise level (delta). The plots 
% compare the Landweber iteration with the Stochastic Approximation 
% with Randomization (SAR) iteration across different noise levels.
%--------------------------------------------------------------------------

clear; close all;

% Common parameters
M = 5;  % Initial M steps to adjust the random initial guess as close as possible to the given initial guess
maxit = 300;  % Maximum number of iterations
delta = 0.02;  % Given noise level
Path = 50;  % Number of samples for SAR iteration
theta = 0.5;  % Randomization level
nel = 256;  % Number of grid points
rand_x0 = 0.01 + 4 * rand(nel, 1);  % Randomly generated initial guess

% Load data for different noise levels and perform SAR and Landweber iterations
%load data001  % Load data for delta = 0.01
delta1 = 0.01; tau1 = 1.8;  
[Error_SAR01, ~, ~, ~, ~, ~, ~] = Func_SAR_delta1(rand_x0, maxit, theta, Path, tau1, M);
[~, ~, Error_Land01] = Func_Land_delta1(rand_x0, maxit, tau1);

%load data003  % Load data for delta = 0.03
delta2 = 0.03; tau2 = 1.3;  
[Error_SAR02, ~, ~, ~, ~, ~, ~] = Func_SAR_delta2(rand_x0, maxit, theta, Path, tau2, M);
[~, ~, Error_Land02] = Func_Land_delta2(rand_x0, maxit, tau2);

%load data005  % Load data for delta = 0.05
delta3 = 0.05; tau3 = 1.2;  
[Error_SAR03, ~, ~, ~, ~, ~, ~] = Func_SAR_delta3(rand_x0, maxit, theta, Path, tau3, M);
[~, ~, Error_Land03] = Func_Land_delta3(rand_x0, maxit, tau3);

% Compute the probability that the error exceeds the noise level delta
Prob_SAR1 = sum(Error_SAR01 > delta1, 2) / Path;
Prob_SAR2 = sum(Error_SAR02 > delta2, 2) / Path;
Prob_SAR3 = sum(Error_SAR03 > delta3, 2) / Path;
Prob_Land1 = sum(Error_Land01 > delta1, 2);
Prob_Land2 = sum(Error_Land02 > delta2, 2);
Prob_Land3 = sum(Error_Land03 > delta3, 2);

% Plot the probabilities for Landweber and SAR methods
figure(1)
h1 = plot(Prob_Land1, 'k', 'LineWidth', 2);  % Plot for Landweber with delta1, delta2, and delta3
hold on
plot(Prob_Land2, 'k', 'LineWidth', 2);  % This line will not be included in the legend
plot(Prob_Land3, 'k', 'LineWidth', 2);  % This line will not be included in the legend
h2 = plot(Prob_SAR1, '--b', 'LineWidth', 2);  % Plot for SAR with delta1
h3 = plot(Prob_SAR2, '--r', 'LineWidth', 2);  % Plot for SAR with delta2
h4 = plot(Prob_SAR3, '--c', 'LineWidth', 2);  % Plot for SAR with delta3
grid on
legend([h1, h2, h3, h4], {'Landweber$\delta_{1,2,3}$', 'SAR$\delta_1$', 'SAR$\delta_2$', 'SAR$\delta_3$'}, 'Interpreter', 'latex')
xlabel('Number of iterations (n)')
t = '$P(||x_n-x^\dag|| \geq \delta_i)$';
title(t, 'Interpreter', 'latex')


figure(2)
h1 = semilogy(Prob_Land1,'k', 'LineWidth',2); 
hold on
semilogy(Prob_Land2,'k', 'LineWidth',2);  % 这一行不会在图例中显示
semilogy(Prob_Land3,'k', 'LineWidth',2);  % 这一行不会在图例中显示
h2 = semilogy(Prob_SAR1,'--b','LineWidth',2);
h3 = semilogy(Prob_SAR2,'--r','LineWidth',2);
h4 = semilogy(Prob_SAR3,'--c','LineWidth',2);
grid on
legend([h1, h2, h3, h4], {'Landweber$-\delta_{1,2,3}$', 'SAR$-\delta_1$', 'SAR$-\delta_2$', 'SAR$-\delta_3$'}, 'Interpreter', 'latex')
xlabel('Number of iterations(n)')
t = '$P(||x_n-x^\dag|| \geq \delta_i)$';
title(t,'interpreter','latex')

ylim([0.0018, 1])  % 这里假设 y 轴范围为 [1e-4, 1]，你可以根据数据进行调整