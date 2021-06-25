function [p, AY] = polynomialPreconditioner(A, v0, deg)

% Compute a polynomial preconditioner for a matrix A.
%
% This is done by solving the normal equations (AV)'AV*y = AV*v,
% where V is a power basis [v A*v ... A^deg*v].
%
% input   A        REAL matrix
%         v0       REAL input vector
%         deg      INTEGER degree of the polynomial preconditioner
%
% output  p        REAL coefficients of polynomial preconditioner

    n = length(A);
    Y = zeros([n deg+1]);
    Y(:, 1) = v0;
    
    for i = 1:deg
        Y(:, i+1) = A*Y(:, i);
    end
    
    AY = A*Y;
    p = linsolve(AY'*AY, AY'*v0);
    p = flip(p);

% END of polynomialPreconditioner.m