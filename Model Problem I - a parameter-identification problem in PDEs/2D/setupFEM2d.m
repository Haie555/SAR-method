function fem = setupFEM2d(n)
% FEM = SETUPFEM2D(NEL)
% setup 2D FEM structure (uniform triangular grid)
% Input:  n   -- number of grid points in each dimension
% Output: fem -- structure containing
%          nel   -- number of elements
%          xx,yy -- nodes
%          h2    -- mesh constant: 2*area(T)
%          A     -- function A(u) returning stiffness matrix (-\Delta + u)
%          M     -- mass matrix for right hand side in V_h
%          Y     -- function Y(y) returning weighted mass matrix
%          P     -- projection V_h to U_h: trapezoidal rule
%          plotU -- plot cell-wise constant functions
%          plotV -- plot piecewise linear (nodal) functions
%
% Christian Clason (christian.clason@uni-graz.at)
% Bangti Jin       (btjin@math.tamu.edu)
% February 27, 2011

%% Set up grid: uniform triangular mesh
x       = linspace(-1,1,n)';     % spatial grid points (uniform in x,y)
h2      = (x(2)-x(1))^2;         % Jacobi determinant of transformation (2*area(T))
[xx,yy] = meshgrid(x);           % coordinates of nodes
nel     = 2*(n-1)^2;             % number of nodes

tri     = zeros(nel,3);          % triangulation
ind = 1;
for i = 1:n-1
    for j = 1:n-1
        node         = (i-1)*n+j+1;              % two triangles meeting at node
        tri(ind,:)   = [node node-1 node+n];     % first triangle (lower left)
        tri(ind+1,:) = [node+n-1 node+n node-1]; % second triangle (upper right)
        ind = ind+2;
    end
end

%% dual triangulation of cell centers (for plotting)
p = zeros(nel,2);               % centroids of triangles
for i = 1:nel
    dex    = tri(i,:);
    p(i,:) = 1/3*[sum(xx(dex)) sum(yy(dex))];
end
pri = delaunay(p(:,1),p(:,2));  % dual triangulation

% plot cell-wise constant on cell centers, linear interpolation
fem.plotU = @(f) trisurf(pri,p(:,1),p(:,2),f);
% plot cell-wise constant on cell centers, linear interpolation
fem.plotV = @(f) surf(xx,yy,reshape(f,n,n));

%% stiffness and mass matrix
[K,M] = assembleFEM(tri,h2);

%% projection on piecewise constants
P0    = @(y) (y(tri(:,1))+y(tri(:,2))+y(tri(:,3)))/3;

%% FEM structure: grid, stiffness & mass matrices, projections etc.
fem.nel = nel;                         % number of elements
fem.xx  = xx;                          % grid
fem.yy  = yy;                          % grid
fem.h2  = h2;                          % mesh constant: 2*area(T)
fem.A   = @(u) K+assembleU(u,tri,h2);  % differential operator
fem.M   = M;                           % mass matrix
fem.Y   = @(y) assembleY(y,tri,h2);    % weighted mass matrix: Y(y)*du = <y du,v>
fem.P   = P0;                          % projection V_h to U_h

end % function setupFEM2D


%% Assemble stiffness, mass matrix
function [K,M] = assembleFEM(t,h2)
nel = size(t,1);    % number of elements

Ke = 1/2*[2 -1 -1 -1 1 0 -1 0 1]'; % elemental stiffness matrix <phi_i',phi_j'>
Me = h2/24 * [2 1 1 1 2 1 1 1 2]'; % elemental mass matrix <phi_i,phi_j>

ent = 9*nel;
row = zeros(ent,1);
col = zeros(ent,1);
valk = zeros(ent,1);
valm = zeros(ent,1);

ind=1;
for el=1:nel
    ll      = ind:(ind+8);         % local node indices
    gl      = t(el,:);             % global node indices
    cg      = gl([1;1;1],:); rg = gl';
    rg      = rg(:,[1 1 1]);
    row(ll) = rg(:);
    col(ll) = cg(:);
    valk(ll)= Ke;
    valm(ll)= Me;
    ind     = ind+9;
end
K = sparse(row,col,valk);
M = sparse(row,col,valm);
end % function assembleFEM


%% Assemble potential mass matrix <u y,v> for u in U_h
function U = assembleU(u,t,h2)
nel = size(t,1);                    % number of elements

Me = h2/24 * [2 1 1 1 2 1 1 1 2]';  % elemental mass matrix <phi_i,phi_j>

ent = 9*nel;
row = zeros(ent,1);
col = zeros(ent,1);
val = zeros(ent,1);

ind=1;
for el=1:nel
    ll      = ind:(ind+8);          % local node indices
    gl      = t(el,:);              % global node indices
    cg      = gl([1;1;1],:); rg = gl';
    rg      = rg(:,[1 1 1]);
    row(ll) = rg(:);
    col(ll) = cg(:);
    val(ll) = u(el)*Me;
    ind     = ind+9;
end
U = sparse(row,col,val);
end % function assembleU


%% Assemble weighted mass matrix: Y(y)*du = <y du,v> for du in U_h
function Y = assembleY(y,t,h2)
nel = size(t,1);                    % number of elements

Me = h2/24 * [2 1 1; 1 2 1; 1 1 2]; % elemental mass matrix <phi_i,phi_j>

ent = 3*nel;
row = zeros(ent,1);
col = zeros(ent,1);
val = zeros(ent,1);

ind=1;
for el=1:nel
    ll      = ind:ind+2;
    dex     = t(el,:)';
    row(ll) = dex;
    col(ll) = [el el el];
    val(ll) = Me*y(dex);            % <y,phi_i>_K_j
    ind     = ind+3;
end
Y = sparse(row,col,val);
end % function assembleY
