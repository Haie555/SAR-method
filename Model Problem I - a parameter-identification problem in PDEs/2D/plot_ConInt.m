clear; clc;

load 2D_data1

% Inversion Parameters
maxit = 120;  
tau = 1.2;  
theta = 0.04;  
Path = 10;

% Initialization
Xt = zeros(nel, Path);
Res = zeros(Path, 1);

for p = 1:Path % Take 'Path' sample paths to calculate the mean
    x0 = ones(nel, 1); % Initial guess
    [S, My, rhs, rrhs, dy, res_y] = forward_2D(fem, x0, Mf, u_delta);
    residual = norm(res_y);
    
    % Store initial error metrics
    RE_sar(1, p) = norm(x0 - x_true) / norm(x_true);
    error_sar(1, p) = norm(x0 - x_true);
    residual_sar(1, p) = residual;
    
    xt = x0;
    k = 1; 
    dt = norm(rhs)^2 / norm(dy)^2;
    
    % Iterative inversion process
    while residual > tau * deltan && k < maxit
        dW = sqrt(dt) * randn(nel, 1); % Wiener process increment
        R = sqrt(mean(residual_sar(:, p)));
        
        [S, My, rhs, rrhs, dy, res_y] = forward_2D(fem, xt, Mf, u_delta);
        dt = norm(rhs)^2 / norm(dy)^2;
        
        % Update solution with stochastic perturbation
        xt = xt + dt * rhs + theta * sqrt(1 / (1 + k * dt)) * R * dW;
        
        % Store error metrics
        RE_sar(k + 1, p) = norm(xt - x_true) / norm(x_true);
        error_sar(k + 1, p) = norm(xt - x_true);
        residual = norm(res_y);
        residual_sar(k + 1, p) = residual;
        
        k = k + 1;
    end
    
    Xt(:, p) = xt; % Store final reconstruction
    Res(p) = norm(xt - x_true); % Store final error
end


mp=zeros(nel,1);     mpp=zeros(nel,1);
MP=zeros(nel,Path);  MPP=zeros(nel,Path);
up_xt=zeros(nel,1);  low_xt=zeros(nel,1);

xt_M=mean(Xt,2);
for i=1:p
    for j=1:nel
        mp(j)=mean(Xt(j,i)-xt_M(j));  % 以重构解的中值为参考点，计算每一个nel上对应的所有Path与之的距离，再排序
        mpp(j)=norm(Xt(j,i)-xt_M(j));
    end
    MP(:,i)=mp;
    MPP(:,i)=mpp;
end
for k=1:nel
    [sort_MP,ind_MP]=sort(MP(k,:));
    up_xt(k)=sort_MP(0.60*Path) + xt_M(k);
    low_xt(k)=sort_MP(0.40*Path) + xt_M(k);
end


% Mean of error metrics across paths
Residual_M_SAR = mean(residual_sar, 2);
RE_M_SAR = mean(RE_sar, 2);
Error_M_SAR = mean(error_sar, 2);

% Plot residual, relative error, and error over iterations
figure(21)
subplot(3, 1, 1)
plot(Residual_M_SAR, 'r'); axis tight; title('Residual');

subplot(3, 1, 2)
plot(RE_M_SAR, 'r'); axis tight; title('Relative Error');

subplot(3, 1, 3)
plot(Error_M_SAR, 'r'); axis tight; title('Error');

% Show reconstruction and confidence bounds
figure(22)
fem.plotU(xt_M); hold on
fem.plotU(up_xt); hold on
fem.plotU(low_xt); hold off
shading interp; camlight headlight; lighting phong; axis tight; drawnow;
