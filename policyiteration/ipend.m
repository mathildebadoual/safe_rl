% Dynamic Programming via Policy Iteration
% Aaron Kandel

clc
clear
close all
%% Define Transition Tables:

% Define Params:
t_max = 20; % [s] Max time
dt1 = 0.025; % [s] Timestep
dt2 = 0.025;
maxit = t_max/(2*dt1);%dt; % iterations


% Battery Parameters:
g = 9.8;
mc = 1;
mp = 0.01;
mt = mp + mc;
l = 0.5;
pml = mp*l;
fm = 10;
p.g = g;
p.mc = mc;
p.mt = mt;
p.mp = mp;
p.pml = pml;
p.fm = fm;
p.l = l;
p.dt1 = dt1;
p.dt2 = dt2;

ns = 5;
nt = 25;
nsd = 5;
ntd = 15;

s = linspace(-2, 2, ns);
t = linspace(-3.14, 3.14, nt);
sd = linspace(-2, 2, nsd);
td = linspace(-2, 2, ntd);

F = linspace(-fm, fm, 10);
xvec = zeros(4,length(s)*length(t)*length(sd)*length(td));

kj = 1;
for i = 1:length(s)
    for j = 1:length(t)
        for k = 1:length(sd)
            for m = 1:length(td)
                xvec(:,kj) = [s(i);t(j);sd(k);td(m)];
                
                kj = kj + 1;  
            end
        end
    end
end


% Assemble table of deterministic state transitions:
% With simple modifications, this code can handle probabilistic state
% transitions from a Markov chain model.  However, for this
% application study the underlying system is deterministic, so the
% transition probabilities are either zero or one.
ds = s(2)-s(1);
dth = t(2)-t(1);
dsd = sd(2)-sd(1);
dtd = td(2)-td(1);
for k = 1:length(F)
    clear Vsim VOC
%     nxvec = A*xvec + B*currentD(k); % Vector of next states
%     % Nearest Neighbor Interpolation:
%     nxvSOC = round(nxvec(1,:)./dsoc) .* dsoc;
%     nxvVRC = round(nxvec(2,:)./dvrc) .* dvrc;
%     nxv = [nxvSOC;nxvVRC];
%     % Compute indices of next states:
    for i = 1:length(xvec)
        [ss,tt,sdsd,tdtd] = env(xvec(1,i), xvec(2,i), xvec(3,i), xvec(4,i), F(k),p);
        nxvec(:,i) = [ss;tt;sdsd;tdtd];
        [~,nis] = min((nxvec(1,i) - s).^2);
        [~,nit] = min((nxvec(2,i) - t).^2);
        [~,nisd] = min((nxvec(3,i) - sd).^2);
        [~,nitd] = min((nxvec(4,i) - td).^2);
        nextInd(i,k) = (nis-1)*nt*nsd*ntd + (nit-1)*nsd*ntd + (nisd-1)*ntd + nitd;
        
    end % END FOR
    
    % Transition cost (SOC reference tracking + constraint penalty):
    tCost(:,k) = (nxvec(2,:)).^2;% + 1000*(Vsim > VsimMax)  )'; 

end % END FOR
%% Policy Iteration:

nInputs = length(F);
nStates = length(nxvec);
discountFactor = 0.998;
optimalPolicy = PolicyIter(nextInd,tCost,nInputs,nStates,discountFactor);
%% Simulate Final Solution:

% Initialize states:
x0 = [0;-pi;0;0]; % 2.5
% x0 = [0;45 * pi/180;0;0];
x = zeros(4,maxit);
x(:,1) = x0;
% Simulate Greedy Policies:
for i = 1:round(maxit)            
    % Find nearest states:
    [~,nis] = min((s - x(1,i)).^2);
    [~,nit] = min((t - x(2,i)).^2);
    [~,nisd] = min((sd - x(3,i)).^2);
    [~,nitd] = min((td - x(4,i)).^2);
    % Compute state index:
    curState = (nis-1)*nt*nsd*ntd + (nit-1)*nsd*ntd + (nisd-1)*ntd + nitd;
    % Use state index with optimal policy lookup table to get optimal input:
    action(i,1) = F(optimalPolicy(curState));

    % Compute State Transitions:
    [a,b,c,d] = env(x(1,i),x(2,i),x(3,i),x(4,i),action(i,1),p);
    x(:,i+1) = [a;b;c;d];

end % END FOR

% PLOT RESULTS:
t2 = 2*dt1*(0:(maxit)); % assign time vector for plotting



figure(1)
subplot(2,1,1)
plot(t2,x(1,:)./max(abs(x(1,:))),t2,x(2,:)./max(abs(x(2,:))),...
    t2,x(3,:)./max(abs(x(3,:))),t2,x(4,:)./max(abs(x(4,:))))
grid on
xlabel('Time [s]')
ylabel('Normalized States')
legend('Position','\theta','Velocity','{\omega}')
title('Initial \theta = 180 degrees')
subplot(2,1,2)
plot(t2(1:end-1),action)
grid on
ylabel('Input [N]')
xlabel('Time [s]')

% figure(1) 
% clf
% subplot(1,5,1)
% hold on
% plot(t2(1:length(action)-1),action(1:end-1))%,'Linewidth',2)
% xlim([0 t_max])
% grid on
% xlabel('Time [s]')
% ylabel('Input [N]')
% subplot(1,5,2)
% hold on
% plot(t2(1:length(x)),x(2,:))
% % plot([0,t2(end)])%,[z_targ, z_targ],'--k')
% grid on
% xlabel('Time [s]')
% ylabel('theta [rad]')
% % legend('Optimal','Target') % 'Eng.'
% % ylim([0 1])
% xlim([0 t_max])
% subplot(1,5,3)
% hold on
% plot(t2(1:length(x)),x(4,:))
% % plot([0 t_max],[VsimMax,VsimMax])%t,V_S)
% % ylim([3 4])
% xlim([0 t_max])
% grid on
% xlabel('Time [s]')
% ylabel('thetaDot [rad/s]')
% % legend('Optimal','Constraint') %
% subplot(1,5,4) 
% plot(t2(1:length(x)),x(1,:))
% grid on
% xlabel('Time [s]')
% ylabel('Position [m]')
% subplot(1,5,5)
% plot(t2(1:length(x)),x(3,:))
% grid on
% xlabel('Time [s]')
% ylabel('Velocity [m/s]')
%% Functions:

function [sn,tn,sdn,tdn] = env(s,t,sd,td,F,p)
 
dt1 = p.dt1;
dt2 = p.dt2;
pml = p.pml;
l = p.l;
mp = p.mp;
mt = p.mt;
g = p.g;

% tau1:
sn1 = s + dt1*sd;
tn1 = t + dt1*td;

ct1 = cos(t);
st1 = sin(t);
temp1 = (F + pml*td*td*st1)/mt;
thacc1 = (g*st1 - ct1*temp1)/(l*( 4/3 - mp*ct1*ct1/mt));
xacc1 = temp1 - pml*thacc1*ct1/mt;
sdn1 = sd + dt1*xacc1;
tdn1 = td + dt1*thacc1;


% tau2:
sn = sn1 + dt2*sdn1;
tn = tn1 + dt2*tdn1;
ct2 = cos(tn1);
st2 = sin(tn1);

temp2 = (F + pml*tdn1*tdn1*st2)/mt;%/(l*(4/3 - mp*ct2*ct2/mt));
thacc2 = (g*st2 - ct2*temp2)/(l*(4/3 - mp*ct2*ct2/mt));
xacc2 = temp2 - pml*thacc2*ct2/mt;

sdn = sdn1 + dt2*xacc2;
tdn = tdn1 + dt2*thacc2;







end



