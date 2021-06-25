function [y, mvps] = applyPolynomial(p, A, v)

% Compute the matrix-vector product p(A)v using Horner's method.
% p(A) is a polynomial of degree d in the matrix A.
%
% If p is an empty array, v is returned unchanged.
%
% input   A        REAL matrix
%         v        REAL input vector
%         p        REAL coefficients [0, 1, ..., d] of degree d polynomial
%
% output  y        REAL result vector
%         mvps     INTEGER matrix-vector products

    np = length(p);
    deg = np-1;
    mvps = 0;

    if (np == 0)
       y = v;
       return
    end

    y = p(1)*A*v + p(2)*v;
    for i = 2:deg
        y = A*y + p(i+1)*v;
    end
    mvps = deg;

% END of applyPolynomial.m
