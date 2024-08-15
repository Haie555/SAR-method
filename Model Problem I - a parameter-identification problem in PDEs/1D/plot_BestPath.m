
%--------------------------------------------------------------------------
% Revised on 2024.08.12
%--------------------------------------------------------------------------

close all; clear, clc

load data002

maxit = 200; M = 5;
tau = 1.3;  delta = 0.02; Path = 500; theta = 0.5;

% Inversion
Xt=zeros(nel,Path);
IterNum=zeros(Path,1);
given_x0 = fem.P(1+0*x); % 给定的初值
rand_x0 = 0.01 + 4 * rand(nel, 1); % 随机生成的初始解

[~, ~, rhs, ~, dy, res_y] = forward_1D(fem, given_x0, Mf, u_delta);
R0 = norm(res_y);

dt = norm(rhs)^2/(norm(dy))^2;
for p=1:Path  % Take 'Path' sample path to calculate the mean
    x0 = rand_x0;
    for k = 1:M
        [~, ~, rhs, ~, dy, res_y] = forward_1D(fem, x0, Mf, u_delta);
        rk = norm(res_y);
        dt = 0.1;
%         dt = 0.5*norm(rhs)^2/(norm(dy))^2;
        z0 = given_x0 - x0 - dt * rhs;
        s1 = sign(z0);
%         dW = z0.* sum(randn(nel),2)/nel;    
        t1 = randn(nel,1);
        dW = z0.* abs(t1);    
        xk = x0 + dt * rhs + dW;
%         X0(:,k+2) = x0;
%         intial_dist(k+1) = norm(x0 - rand_x0);
%         figure(1)
%         plot(intial_dist);
        Residual_SAR(k,p) = rk;
        RE_SAR(k,p) = norm(xk-x_true)/norm(x_true);
        x0 = xk;
    end
    
    xt = x0;
    [~, ~, ~, ~, ~, res_y] = forward_1D(fem, xt, Mf, u_delta);
    Residual = norm(res_y);
%     R0 = Residual;
    k = M+1;  % 从第 M+1 步开始
    while Residual > tau*deltau && k < maxit
        t2 = randn(nel,1);
        dW = sqrt(dt)*abs(t2);
        [S, My, rhs, rrhs ,dy, res_y] = forward_1D(fem,xt,Mf,u_delta);
        dt = norm(rhs)^2/(norm(dy))^2;
        R = min(norm(res_y),R0);
        xt = xt+dt*rhs+delta*theta*sqrt(1/(1+dt))*R*dW;
        RE_SAR(k,p) = norm(xt-x_true)/norm(x_true);
        Error_SAR(k,p)= norm(xt-x_true);
        Residual = norm(res_y);
        Residual_SAR(k,p) = Residual;
        k = k+1;
    end
    IterNum(p) = k;
    Xt(:,p)=xt;

end

% Expection of Reconstruction 
xt_M = mean(Xt,2);

% Best-Path Reconstruction by Minmum Norm Solution
BPR1 = vecnorm(Xt - x_true); % 计算 Xt 中每个列向量与 x_true 之间的 2 范数距离
[V1,P1]=min(BPR1); % 找到最小距离以及对应的列向量的位置
BPR_MinNorm=Xt(:,P1); % 找到距离 x_true 最近的向量
MinNorm = V1
MeanNorm = mean(BPR1)

% Best-Path Reconstruction by Minmum Residula Solution
BPR2=zeros(Path,1);
for p=1:Path
    [S, My, rhs, rrhs ,dy, res_y] = forward_1D(fem,Xt(:,p),Mf,u_delta);
    BPR2(p) = norm(res_y);
end
[V2,P2]=min(BPR2);
BPR_MinRes=Xt(:,P2);
MinRes = V2;

% Expection of Reconstruction 
xt_M = mean(Xt,2);
[ZZ,MZ] = min(RE_SAR);
[BPValue,BestPath] = min(MZ);

t1 = RE_SAR(:,BestPath);
t2 = find(t1~=0);
t3 = length(t2);
RE_BP_SAR = t1(1:t3);

r1 = Residual_SAR(:,BestPath);
r2 = find(r1~=0);
r3 = length(r2);
Res_BP_SAR = r1(1:r3);

% plot Expection of Residual
figure(1)
plot(Res_BP_SAR,'r'); title('Residual of BestPath');

% plot Expection of Error
figure(2)
plot(RE_BP_SAR,'r'); title('Relative error of BestPath');

% 绘制基础图形
figure(3)
plot(xc,x_true,'k', 'LineWidth',1);
% hold on
% plot(xc,xt_M,'r', 'LineWidth',1);
hold on
plot(xc,BPR_MinNorm,'r', 'LineWidth',1);
% hold on
% plot(xc, BPR_MinRes, 'b', 'LineWidth',1);
set(gca,'XTick',-1:0.2:1);  % 设置x轴刻度
% % plot(xc,x_true,'k', xc,xt_M,'--m', xc,low_xt,'--b', xc,up_xt,'--r');
% legend('Exact solution', 'Expectation of SAR', 'BPR-MinNorm', 'BPR-MinRes','Location','south');
legend('Exact solution',  'Approximate solution by SAR', 'Location','south');
% legend('Exact solution', 'Expectation of SAR')
axis tight; % 自动调整坐标轴范围以适合数据

% % plot 局部放大图
% % 创建子图坐标轴 
% hold on
% axes('Position',[0.4 0.4 0.25 0.25]) % 前两个量表示子图坐标轴的初始位置，后两个表示坐标轴的长度和宽度
% % 在子坐标轴中绘制同样的数据 
% plot(xc,x_true,'k', 'LineWidth',1.2);
% hold on
% plot(xc,xt_M,'r', 'LineWidth',1.2);
% legend('Exact solution', 'Expectation of SAR');
% % hold on
% % plot(xc,BPR_MinNorm,'c', 'LineWidth',1.2);
% % hold on
% % plot(xc, BPR_MinRes, 'b', 'LineWidth',1.2);
% % hold on
% % axis([-0.15 0.15 2.95 3.02]) %子图在原图中截取的横纵位置
% % grid on
% % set(gca,'YTick',2.95:0.03:3.02);
% drawnow

MeanIterNum = mean(IterNum) % 迭代步数的均值
MinIterNum = min(IterNum)  % 所有Path中最小迭代步数

Sort_IterNum = sort(IterNum);
MedIterNum = median(Sort_IterNum) % 迭代步数的中位数

norms = sqrt(sum((Xt - x_true).^2, 1)); % 计算 Xt 每一列与 真解x_true 的欧几里得范数
normalized_norms = norms / norm(x_true);
threshold = 0.1;
probability = sum(normalized_norms < threshold) / size(Xt, 2);
% 输出概率
disp(['相对误差小于给定 threshold 的概率: ', num2str(probability)]);



