%--------------------------------------------------------------------------
% Revised on 2024.08.13
%--------------------------------------------------------------------------

close all; clear; clc

% Load data
load data002  % Load dataset for delta = 0.02

% Parameters
M = 5;  % Number of initial iterations using the algorithm in the paper to adjust the random initial guess
maxit = 200;  % Maximum number of iterations
delta = 0.02;  % Given noise level
Path = 50;  % Number of samples for SAR iteration
theta = 0.0;  % Randomization level
tau = 1.3;  % Stopping criterion parameter
rand_x0 = 0.01 + 4 * rand(nel, 1);  % Randomly generated initial guess

% Landweber iteration
[RE_Land, Res_Land, Error_Land] = Func_Land(rand_x0, maxit, tau);
C0 = Error_Land(end);  % Final error of the Landweber iteration

% SAR iteration
[Error_SAR, BPR1, RE_BP_SAR, Res_BP_SAR, IterNum, MeanNorm, MinNorm] = Func_SAR(rand_x0, maxit, theta, Path, tau, M);

% Calculate probabilities of achieving a fraction of the final Landweber error
P11 = length(find(BPR1 <= 0.8*C0)) / Path;  % Probability of achieving 80% of Landweber iteration
P10 = length(find(BPR1 <= 0.5*C0)) / Path;  % Probability of achieving 50% of Landweber iteration
P09 = length(find(BPR1 <= 0.3*C0)) / Path;  % Probability of achieving 30% of Landweber iteration
P08 = length(find(BPR1 <= 0.1*C0)) / Path;  % Probability of achieving 10% of Landweber iteration

% Plot residuals of SAR and Landweber iterations
figure(1)
semilogy(Res_BP_SAR, 'r', 'LineWidth', 2);  % Plot residuals of SAR iteration
hold on
semilogy(Res_Land, 'k', 'LineWidth', 2);  % Plot residuals of Landweber iteration
title('Residual');
xlabel('Number of iterations (n)');
legend('SAR', 'Landweber');

% Plot relative errors of SAR and Landweber iterations
figure(2)
plot(RE_BP_SAR, 'r', 'LineWidth', 2);  % Plot relative errors of SAR iteration
hold on
plot(RE_Land, 'k', 'LineWidth', 2);  % Plot relative errors of Landweber iteration
title('Relative Error');
xlabel('Number of iterations (n)');
legend('SAR', 'Landweber');
