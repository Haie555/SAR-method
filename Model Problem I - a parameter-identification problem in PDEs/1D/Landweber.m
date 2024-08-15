%--------------------------------------------------------------------------
% Revised on 2024.08.13
%--------------------------------------------------------------------------

clear; close all

load data002

maxit = 500;
tau = 1.3;
x0 = 0.01 + 4* rand(nel, 1); % 初始解，随机生成
[S, My, rhs, rrhs ,dy, res_y] = forward_1D(fem,x0,Mf,u_delta);
Residual = norm(res_y);
xt = x0;

k  = 1;
while Residual> tau*deltau && k<maxit
    
    RE_Land(k) = norm(xt-x_true)/norm(x_true);
    Error_Land(k) = norm(xt-x_true);
    Residual = norm(res_y);
    Residual_Land(k) = Residual;
    [S, My, rhs, rrhs ,dy, res_y] = forward_1D(fem,xt,Mf,u_delta);
    dt = norm(rhs)^2/(norm(dy))^2;
    xt = xt+dt*rhs;
    k = k+1;
end
plotx = xt;
Error_Land = Error_Land';
RE_Land = RE_Land';
Residual_Land = Residual_Land';


% plot
figure(1)
plot(Residual_Land,'r'); title('Residual');
%
figure(2)
plot(RE_Land,'r'); title('Relative error');

figure(3)
plot(Error_Land,'r'); title('Error');

figure(4)
plot(xc,x_true,'k', xc, plotx,'r', 'LineWidth',1);
axis tight;
legend('Exact solution', 'Landweber');

