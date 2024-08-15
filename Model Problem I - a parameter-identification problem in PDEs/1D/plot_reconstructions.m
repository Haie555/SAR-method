%--------------------------------------------------------------------------
% Revised on 2024.08.13
%--------------------------------------------------------------------------

close all; clear, clc

load data002

maxit = 200;
tau = 1.01;  delta = 0.02; Path = 500; theta = 0.05;

% Inversion
Xt=zeros(nel,Path);
IterNum=zeros(Path,1);
new_Xt=zeros(nel,Path);
x0 = fem.P(1+0*x);

for p=1:Path  % Take 'Path' sample path to calculate the mean

    [S, My, rhs, rrhs ,dy, res_y] = forward_1D(fem,x0,Mf,u_delta);
    residual = norm(res_y);
    RE_sar(1,p) = norm(x0-x_true)/norm(x_true);
    error_sar(1,p) = norm(x0-x_true);
    residual_sar(1,p) = residual;
    xt = x0;
    R = sqrt(mean(residual_sar(:,p)));
    dt = norm(rhs)^2/(norm(dy))^2;
    
    k = 1;
    while residual>tau*deltau && k<maxit
        dW = sqrt(dt)*sum(randn(nel),2)/nel;
        [S, My, rhs, rrhs ,dy, res_y] = forward_1D(fem,xt,Mf,u_delta);
        dt = norm(rhs)^2/(norm(dy))^2;
        xt = xt+dt*rhs+theta*sqrt(1/(1+dt))*R*dW;
        RE_sar(k+1,p) = norm(xt-x_true)/norm(x_true);
        error_sar(k+1,p)= norm(xt-x_true);
        residual = norm(res_y);
        residual_sar(k+1,p) = residual;
        k = k+1;
    end
    IterNum(p) = k;
    Xt(:,p)=xt;

end


% Best-Path Reconstruction by Minmum Norm Solution
BPR1=zeros(Path,1);
for p=1:Path
    BPR1(p)=norm(Xt(:,p)- x_true);
end
[V1,P1]=min(BPR1);
BPR_MinNorm=Xt(:,P1);
MinNorm = V1
RE_MinNorm = RE_sar(:,P1);

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
% MeanNorm = norm(xt_M-x_true)
MeanNorm = mean(BPR1)

Residual_M_SAR=mean(residual_sar,2);
RE_M_SAR=mean(RE_sar,2);
Error_M_SAR=mean(error_sar,2);

% plot Expection of Residual
figure(1)
plot(Residual_M_SAR,'r'); title('residual');

% plot Expection of Error
figure(2)
plot(Error_M_SAR,'r'); title('error');

% 绘制基础图形
figure(3)
plot(xc,x_true,'k', 'LineWidth',1);
hold on
plot(xc,xt_M,'r', 'LineWidth',1);
hold on
plot(xc,BPR_MinNorm,'c', 'LineWidth',1);
hold on
plot(xc, BPR_MinRes, 'b', 'LineWidth',1);
set(gca,'XTick',-1:0.2:1);  % 设置x轴刻度
% plot(xc,x_true,'k', xc,xt_M,'--m', xc,low_xt,'--b', xc,up_xt,'--r');
legend('Exact solution', 'Expectation of SAR', 'BPR-MinNorm', 'BPR-MinRes','Location','south');
axis tight; % 自动调整坐标轴范围以适合数据

% plot 局部放大图
% 创建子图坐标轴 
hold on
axes('Position',[0.4 0.4 0.25 0.25]) % 前两个量表示子图坐标轴的初始位置，后两个表示坐标轴的长度和宽度
% 在子坐标轴中绘制同样的数据 
plot(xc,x_true,'k', 'LineWidth',1.2);
hold on
plot(xc,xt_M,'r', 'LineWidth',1.2);
hold on
plot(xc,BPR_MinNorm,'c', 'LineWidth',1.2);
hold on
plot(xc, BPR_MinRes, 'b', 'LineWidth',1.2);
hold on
axis([-0.15 0.15 2.95 3.02]) %子图在原图中截取的横纵位置
grid on
set(gca,'YTick',2.95:0.03:3.02);
drawnow


