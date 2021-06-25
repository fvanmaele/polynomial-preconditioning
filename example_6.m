%% SHERMAN5
[A,rows,cols,entries,rep,field,symm] = mmread("sherman5.mtx"); % sparse format
n = 3312;

%% Input parameters
x0 = zeros(length(A), 1);
b = normrnd(0, 1, [n,1]);
restart = 100;
max_it = 200;
tol = 1e-8;

%%
[x, error, iter, flag, resvec, spmvs, inps] = gmresArnoldi(A, x0, b, [], restart, max_it, tol);

%% Polynomial preconditioning
%seed = 42;
%seed = 1892;
seed = 666;
rng(seed)
%v0 = normrnd(0, 1, [n,1]);
v0 = unifrnd(-1, 1, [n,1]);
%v0 = b;
p = cell(10, 1);
for deg = 1:15
    p{deg} = polynomialPreconditioner(A,v0,deg);
end

%% Solve preconditioned system
[x_p1, error_p1, iter_p1, flag_p1, resvec_p1, spmvs_1, inps_1] = gmresArnoldi(A, x0, b, p{1}, restart, max_it, tol);
[x_p2, error_p2, iter_p2, flag_p2, resvec_p2, spmvs_2, inps_2] = gmresArnoldi(A, x0, b, p{2}, restart, max_it, tol);
[x_p3, error_p3, iter_p3, flag_p3, resvec_p3, spmvs_3, inps_3] = gmresArnoldi(A, x0, b, p{3}, restart, max_it, tol);
[x_p4, error_p4, iter_p4, flag_p4, resvec_p4, spmvs_4, inps_4] = gmresArnoldi(A, x0, b, p{4}, restart, max_it, tol);
[x_p5, error_p5, iter_p5, flag_p5, resvec_p5, spmvs_5, inps_5] = gmresArnoldi(A, x0, b, p{5}, restart, max_it, tol);
[x_p6, error_p6, iter_p6, flag_p6, resvec_p6, spmvs_6, inps_6] = gmresArnoldi(A, x0, b, p{6}, restart, max_it, tol);
[x_p7, error_p7, iter_p7, flag_p7, resvec_p7, spmvs_7, inps_7] = gmresArnoldi(A, x0, b, p{7}, restart, max_it, tol);
[x_p8, error_p8, iter_p8, flag_p8, resvec_p8, spmvs_8, inps_8] = gmresArnoldi(A, x0, b, p{8}, restart, max_it, tol);
[x_p9, error_p9, iter_p9, flag_p9, resvec_p9, spmvs_9, inps_9] = gmresArnoldi(A, x0, b, p{9}, restart, max_it, tol);
[x_p10, error_p10, iter_p10, flag_p10, resvec_p10, spmvs_10, inps_10] = gmresArnoldi(A, x0, b, p{10}, restart, max_it, tol);
[x_p11, error_p11, iter_p11, flag_p11, resvec_p11, spmvs_11, inps_11] = gmresArnoldi(A, x0, b, p{11}, restart, max_it, tol);
[x_p12, error_p12, iter_p12, flag_p12, resvec_p12, spmvs_12, inps_12] = gmresArnoldi(A, x0, b, p{12}, restart, max_it, tol);
[x_p13, error_p13, iter_p13, flag_p13, resvec_p13, spmvs_13, inps_13] = gmresArnoldi(A, x0, b, p{13}, restart, max_it, tol);
[x_p14, error_p14, iter_p14, flag_p14, resvec_p14, spmvs_14, inps_14] = gmresArnoldi(A, x0, b, p{14}, restart, max_it, tol);
[x_p15, error_p15, iter_p15, flag_p15, resvec_p15, spmvs_15, inps_15] = gmresArnoldi(A, x0, b, p{15}, restart, max_it, tol);

%%
ax = axes;
plot(resvec,'LineWidth',1.5)
hold on
%plot(resvec_p1,'LineWidth',1.5)
%plot(resvec_p2,'LineWidth',1.5)
plot(resvec_p3,'LineWidth',1.5)
%plot(resvec_p4,'LineWidth',1.5)
plot(resvec_p5,'LineWidth',1.5)
%plot(resvec_p6,'LineWidth',1.5)
plot(resvec_p7,'LineWidth',1.5)
%plot(resvec_p8,'LineWidth',1.5)
%plot(resvec_p9,'LineWidth',1.5)
plot(resvec_p10,'LineWidth',1.5)
%plot(resvec_p11,'LineWidth',1.5)
plot(resvec_p12,'LineWidth',1.5)
%plot(resvec_p13,'LineWidth',1.5)
%plot(resvec_p14,'LineWidth',1.5)
plot(resvec_p15,'LineWidth',1.5)
ax.YScale = 'log';
xlabel('Iterations')
ylabel('Residual')
ylim([resvec_p10(iter_p10) resvec(1)])
legend('deg = 0','deg = 3', 'deg = 5', 'deg = 7', 'deg = 10','deg = 12', 'deg = 15')

%% Eigenvalues
specA = eig(full(A));
specPA = cell(15, 1);
for deg = 1:15
   specPA{deg} = polyval([p{deg};0], specA);
end

%%
%plot(specA, 'o')
%hold on
%plot(specPA{3}, 'o')
%plot(specPA{5}, 'o')
plot(specPA{7}, 'o')
%plot(specPA{10}, 'o')
%axis([-0.5 2.5 -1.5 1.5])
%plot(specPA{15}, 'o')

%% Preconditioned matrix
PA = cell(15, 1);
for deg = 1:15
    PA{deg} = polyvalm([p{deg};0], A);
end

%% Condition number (estimate)
condA = condest(A);
condPA = cell(15, 1);
for deg = 1:15
    condPA{deg} = condest(PA{deg});
end

%%
ax = axes;
plot([0:15], [condA condPA{1} condPA{2} condPA{3} condPA{4} condPA{5} condPA{6} condPA{7} condPA{8} condPA{9} condPA{10} condPA{11} condPA{12} condPA{13} condPA{14} condPA{15}], 'x-')
xlabel('Polynomial degree')
ylabel('Condition number (estimate)')
