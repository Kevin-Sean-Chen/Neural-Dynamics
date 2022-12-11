%%%Analytic_adaptive_subunits

tt = 0:0.01:10;
ps = 0.2:0.1:1;
tau = 0.5;
fit_taus = zeros(1,length(ps));

for pp = 1:length(ps)
    temp = (1+exp(-(1+ps(pp))*tt/tau))/(1+ps(pp));
    at = temp/max(temp);
    plot(at)
    hold on
    f = @(b,x) b(1).*exp(b(2).*x)+b(3);                                     % Objective Function
    B = fminsearch(@(b) norm(at - f(b,tt)), [1 ps(pp) 0.5]);                % Estimate Parameters
    
    fit_taus(pp) = abs(B(2));
end
xlabel('time (s)')
ylabel('adaptive gain a(t)')

figure()
plot(ps, fit_taus, '-o')
xlabel('input frequency p_f')
ylabel('decay \tau_{fit}')