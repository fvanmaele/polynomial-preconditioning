% Solve model problem for n = 5, 10, 20, 50, 100
% Seminar: Analysis of Polynomial Preconditioners
% Author: Ferdinand Vanmaele

%% Discretization
dim = [5 10 20 50 100];
h = 1./(dim+1);
x = cell(length(dim), 1);
for k = 1:length(dim)
   x{k} = linspace(0, 1, dim(k)+2);
end

%% f(x) = x^2
f = cell(length(dim), 1);
for k = 1:length(dim)
    kmax = length(x{k})-1;
    f{k} = x{k}(2:kmax).^2; % defined on (0, 1)
end
clear kmax

%% Exact solution
u_exact = @(x) -x.^4./12 + x./12;

%% Central difference approximation
A = cell(length(dim), 1);
for k = 1:length(dim)
    n = dim(k);
    A{k} = diag(2*ones([n, 1]), 0) + diag(-1*ones([n-1, 1]), -1) + diag(-1*ones([n-1, 1]), 1);
    A{k} = 1/(h(k).^2) * A{k};
end
clear n

%% Solve linear system
u = cell(length(dim), 1);
for k = 1:length(dim)
   u{k} = A{k} \ f{k}';
   % add boundary condition
   u{k} = [0; u{k}; 0];
end

%% Polynomial preconditioning (n = 5, deg = 2)
v0 = [1/3; -1; 0; -1; -1];
[p2, AY2] = polynomialPreconditioner(A{1}, v0, 2);
[p5, AY5] = polynomialPreconditioner(A{1}, v0, 5);
Ainv = polyvalm(p5,A{1})*A{1};

%% Plots
plot(x{1}, u{1}, '-*')
hold on
%plot(x{2}, u{2}, '-x')
plot(x{3}, u{3}, '-o')
%plot(x{4}, u{4})
plot(x{5}, u_exact(x{5}), 'g')
xticks(x{1})
legend('n = 5', 'n = 20', 'Exact solution')
ylim([0.032 0.040])