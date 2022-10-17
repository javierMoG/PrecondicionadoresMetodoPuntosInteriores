function [x, lambda, mu, z, norma, cinter, mediaIterK, totIterK] = IntPointPrecondC(Q,A,F,d,b,c)
% Método de puntos interiores con precondicionador para el problema 
% cuadrático
% Min   (0.5)* x' * Q * x + d´* x
% s.a.   A * x = b
%        F*x >= c

% Familia de precondicionadores spd (Eva K. Lee y Zeferino Parada)
% k=-1

% In
% Q.- matriz nxn simétrica y positiva definida
% A.- matriz mxn con m <= n y rango(A) = m.
% F.- matriz pxn.
% d.- vector columna en R^n .
% b.- vector columna en R^m .
% c.- vector columna en R^p .
%
%Out
% x.- vector en R^n con la aproximación del mínimo local.
% lambda.- vector en R^m con la aproximación al multiplicador de Lagrange 
%          asociado a la restricción de igualdad.
% mu.- vector en R^p con la aproximación al multiplicador de Lagrange 
%      asociado a la restricción de desigualdad.
% z.- vector en R^p con la variable de holgura en la restricción de 
%     desigualdad.
% norma .- el valor de la norma euclidiana de las condiciones de KKT en la
%          en la última iteración
% cinter .- número de iteraciones del método de puntos interiores
% mediaIterK .- promedio de iteraciones del método de minres por cada paso
%               del método de puntos interiores
% totIterK .- es la suma del total de iteraciones de minres realizadas en
%             cada uno de los pasos

% Parámetros iniciales
tol = 1e-06;
maxiter = 100;
cinter = 0;
kIter = [];
%-----------------------------------------------------------
n = length(d);
m = length(b);
p = length(c);
%-----------------------------------------------------------
x = ones(n,1);
lambda = zeros(m,1);
mu = ones(p,1);
z = F*x - c + (0.5)*ones(p,1);
tau = (0.5)*(mu'*z)/p;
%-----------------------------------------------------------
% Norma de las condiciones necesarias de KKT
H =[Q*x+A'*lambda-F'*mu+d; A*x-b;F*x-z-c;mu.*z];
norma = norm(H);
condK = 1;
disp('Iter      CNPO            tau     rcond(K)')
disp('-----------------------------------------')
while(norma > tol && cinter < maxiter  && condK > eps)
    % Resuelve el sistema lineal de Newton para la trayectoria central
    D = diag(mu./z);
    G = Q+F'*D*F;
    w = zeros(p,1);
    for k = 1:p
        w(k) = F(k,:)*x -c(k)-(tau/mu(k));
    end
    dg = Q*x+A'*lambda-F'*mu+d+F'*D*w;

    K = [G  A'; A  zeros(m)]; 
    condK = rcond(K); 
    ld = -[dg ; A*x-b];

    % Precondicionador
    %-------------------------------------------------------
    % Escalamiento de la matriz de pesos
    par = norm(A,'inf')^2/norm(G,'inf');
    % Bloque (1,1) del precondicionador
    K11 = G +(1/par)*(A'*A);
    % Bloque (2,2) del precondicionador
    K22 = par*eye(m);
    % Factorización de Cholesky del bloque (1,1)
    L11 = chol(K11)';
    % Factorización de Cholesky del bloque (2,2)
    L22 = sqrt(par)*eye(m);
    % Inversa de L22
    L22inv = 1/sqrt(par)*eye(m);
    % Bloque (1,2) de la factorización de Cholesky del precondicionador
    L12 = -A'*L22inv;
     %------------------------------------------------------
    % Factorización de Cholesky del precondicionador
    LP = [L11 L12; zeros(m,n) L22];
    % Sistema Lineal a resolver
    [y,flag,relres,mIter] = minres(K,ld,1.e-05,100,LP,LP');
    kIter= [kIter mIter];
    %-------------------------------------------------------
    % Se calculan los pasos
    Dx = y(1:n);
    Dlambda = y(n+1:n+m);
    
    Dmu =-(D)*(F*Dx+w);
    Dz = -( (1./mu).*(z.*Dmu - tau) + z ); 
    %---------------------------------------------------------- 
    % Acorta el paso
    bt = ones(p,1); gm = ones(p,1);
    for k = 1:p
        if(Dmu(k) < 0)
            bt(k) = -(mu(k)/Dmu(k));
        end
        if(Dz(k) < 0)
            gm(k) =-(z(k)/Dz(k));
        end 
    end

    alfa =(0.995)*min([1 ; bt;gm ]);    
    %---------------------------------------------------------
    % Nuevo punto
    x = x + alfa*Dx;
    lambda = lambda + alfa*Dlambda;
    mu = mu + alfa*Dmu;
    z  = z + alfa*Dz;
    %---------------------------------------------------------  
    % Nueva tau
    tau = (0.5)*(mu'*z)/p;
    %---------------------------------------------------------  
    %Condiciones necesarias de primer orden
    H=[Q*x+A'*lambda-F'*mu+d;A*x-b; -F*x+z+c;mu.*z];
    norma = norm(H);
    cinter = cinter + 1;
    disp(sprintf('%3.0f  %2.8f  %2.8f %2.16f',cinter,norma,2*tau, condK))
end

mediaIterK=mean(kIter);
totIterK=sum(kIter);
end

        
