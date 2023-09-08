function [x, lambda, mu, z, norma, cinter] = IntPoint(Q,A,F,d,b,c)
% M�todo de puntos interiores para el problema cuadr�tico
% Min   (0.5)* x' * Q * x + d�* x
% s.a.   A * x = b
%        F*x >= c

% Se resuelve el Sistema de Netwon Lineal Reducido

% In
% Q.- matriz nxn sim�trica y positiva definida
% A.- matriz mxn con m <= n y rango(A) = m.
% F.- matriz pxn.
% d.- vector columna en R^n .
% b.- vector columna en R^m .
% c.- vector columna en R^p .
%
%Out
% x.- vector en R^n con la aproximaci�n del m�nimo local.
% lambda.- vector en R^m con la aproximaci�n al multiplicador de Lagrange 
%          asociado a la restricci�n de igualdad.
% mu.- vector en R^p con la aproximaci�n al multiplicador de Lagrange 
%      asociado a la restricci�n de desigualdad.
% z.- vector en R^p con la variable de holgura en la restricci�n de 
%     desigualdad.
% norma .- el valor de la norma euclidiana de las condiciones de KKT en la
%          en la �ltima iteraci�n
% cinter .- n�mero de iteraciones del m�todo de puntos interiores

% Par�metros iniciales
tol = 1e-06;
maxiter = 100;
cinter = 0;
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
H =[Q*x+A'*lambda-F'*mu+d; A*x-b;F*x-z-c; mu.*z];
norma = norm(H, "inf");
disp('Iter      CNPO     alpha      tau     rcond(K)')
disp('-----------------------------------------')
while(norma > tol && cinter < maxiter)
    % Resuelve el sistema lineal de Newton para la trayectoria central
    D = diag(mu./z);
    G = Q+F'*D*F;
    w = zeros(p,1);
    for k = 1:p
        w(k) = F(k,:)*x -c(k)-(tau/mu(k));
    end
    dg = Q*x+A'*lambda-F'*mu+d+F'*D*w;
    
    K = [G  A'; A  zeros(m)];
    % Calculamos el condicional de la matriz G
    condK = rcond(G);
    ld = -[dg ; A*x-b];
    % Sistema Lineal a resolver
    y = K \ ld;
    %--------------------------------------------------------
    % Se calculan los pasos
    Dx = y(1:n);
    Dlambda = y(n+1:n+m);
    
    Dmu =-(D)*(F*Dx+w);
    Dz = -( (1./mu).*(z.*Dmu - tau) + z );
   %---------------------------------------------------------- 
    % Recorte del paso
    bt = ones(p,1); gm = ones(p,1);
    for k = 1:p
        if(Dmu(k) < 0)
            bt(k) = -(mu(k)/Dmu(k));
        end
        if(Dz(k) < 0)
            gm(k) =-(z(k)/Dz(k));
        end 
    end
   
    alfa =(0.995)*min([1 ; bt; gm]);  
    %-----------------------------------------------------------
    % Nuevo punto
       x = x + alfa*Dx;
       lambda = lambda + alfa*Dlambda;
       mu = mu + alfa*Dmu;
       z  = z + alfa*Dz;
     %----------------------------------------------------------  
     % Nueva tau
        tau = (0.5)*(mu'*z)/p;
     %----------------------------------------------------------  
       %Condiciones necesarias de primer KKT
       H=[Q*x+A'*lambda-F'*mu+d;A*x-b; -F*x+z+c;mu.*z];
       norma = norm(H, "inf");
       cinter = cinter + 1;
       fprintf('%3.0f  %2.8f  %2.8f %2.8f %2.16f\n',cinter,norma,alfa, 2*tau, condK)
end

end
        
