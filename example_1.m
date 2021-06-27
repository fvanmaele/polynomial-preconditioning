%% Initialize linear system
% 0, 2, 4, ... 1998
n = 2000;
A = [];
for i = 0:2:1998
   alpha = i*pi/n;
   a = 1 + cos(alpha);
   b = sin(alpha);
   c = -sin(alpha);
   d = 1 + cos(alpha);
   block = [a b; c d];
   A = blkdiag(A, block);
end
A = sparse(A); % CSC format

%%
seed = 42;
rng(seed);

%% Input parameters
x0 = zeros(length(A), 1);
b = normrnd(0, 1, [n,1]);
restart = 100;
max_it = 200;
tol = 1e-8;

%%
[x, error, iter, flag, resvec, spmvs, inps] = gmresArnoldi(A, x0, b, [], restart, max_it, tol);

%% Polynomial preconditioning
v0 = unifrnd(-1, 1, [n,1]);
p = cell(10, 1);
for deg = 1:10
    p{deg} = polynomialPreconditioner(A,v0,deg);
end

%% Solve preconditioned system (deg 3, 5, 7, 10)
[x_p3, error_p3, iter_p3, flag_p3, resvec_p3, spmvs_3, inps_3] = gmresArnoldi_right_precond(A, x0, b, p{3}, restart, max_it, tol);
[x_p5, error_p5, iter_p5, flag_p5, resvec_p5, spmvs_5, inps_5] = gmresArnoldi_right_precond(A, x0, b, p{5}, restart, max_it, tol);
[x_p7, error_p7, iter_p7, flag_p7, resvec_p7, spmvs_7, inps_7] = gmresArnoldi_right_precond(A, x0, b, p{7}, restart, max_it, tol);
[x_p10, error_p10, iter_p10, flag_p10, resvec_p10, spmvs_10, inps_10] = gmresArnoldi_right_precond(A, x0, b, p{10}, restart, max_it, tol);

%%
ax = axes;
plot(resvec,'LineWidth',1.5)
hold on
plot(resvec_p3,'LineWidth',1.5)
plot(resvec_p5,'LineWidth',1.5)
plot(resvec_p7,'LineWidth',1.5)
plot(resvec_p10,'LineWidth',1.5)
ax.YScale = 'log';
xlabel('Iterations')
ylabel('Residual')
ylim([resvec_p10(iter_p10) resvec(1)])
legend('deg = 0','deg = 3','deg = 5','deg = 7','deg = 10')

%% Eigenvalues
specA = eig(full(A));
specPA = cell(10, 1);
for deg = 1:10
   specPA{deg} = sparse(polyval([p{deg};0], specA));
end

%%
plot(specA, 'o')
hold on
%plot(specPA{3}, 'o')
%plot(specPA{5}, 'o')
%plot(specPA{7}, 'o')
plot(specPA{10}, 'o')
%axis([-0.5 2.5 -1.5 1.5])

%% Condition number (estimate)
condA = condest(A);
condPA = cell(10, 1);
for deg = 1:10
    condPA{deg} = condest(polyvalm([p{deg};0], A));
end

%%
ax = axes;
plot([0:10], [condA condPA{1} condPA{2} condPA{3} condPA{4} condPA{5} condPA{6} condPA{7} condPA{8} condPA{9} condPA{10}], 'x-')
xlabel('Polynomial degree')
ylabel('Condition number (estimate)')