function [ S, My, rhs, rrhs ,dy, res_y ] = forward_2D(fem,u0,Mf,yd)
% Input:  fem    -- finite element structure containing matrices
%                   (A(u),M,Y) and projection operator (P)
%         Mf     -- right hand side with mass matrix already applied
%         yd     -- measured data
%         ue     -- exact solution (to evaluate reconstruction error)

% Output: rhs , rrhs     

%% Setup parameters
P = fem.P;      % Projection onto piecewise constants
M = fem.M;      % Mass matrix for piecewise constant right hand side
Y = fem.Y;      % Y(v)*du assembles weighted mass matrix for piecewise constant du and linear v

%% loop

% initialize iterates

    A = fem.A(u0);                     % differential operator
    try
        R = chol(A); Rt=R';           % precompute Cholesky factors; faster in 2D
        S = @(f) R\(Rt\f);            % (linearized) solution operator
    catch notspd %#ok<NASGU>
        S = @(f) A\f;                 % fallback if numerically semidefinite
    end
    
    y = S(Mf);                        % state
    res_y = y-yd;                         % residual
    
    My = Y(y);
    q  = S(-(M*res_y));                   % adjoint
     
    % gradient
    
    rhs  = - P(y.*q);                   % rhs = S'(uk)'*(S(uk)-y_delta)

    dy = S(-My*rhs);                  % differential change in state
                                      % dy = S'(uk)*rhs
                                     
    ddy = S(-(M*dy));
    
    rrhs = P(y.*ddy);                % rrhs = S'(uk)'*dy = S'(uk)'*S'(uk)*S'(uk)'*(S(uk)-y_delta)

    

  
    
end 

