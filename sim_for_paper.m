% simulation script as used in Figure 2 of main paper 

n=50;
steps=200;  % time-steps in one "iteration". each time-step is ~ 1 burst duration.


beta = .25;	% global inhibition strength
wmax=1;		% single synapse hard bound
m=1;		% Wmax = m*wmax
pn=2;		 
p=pn/n;		% probability of external stimulation of any neuron at any time
eta=0.125;	% learning rate parameter	
epsilon=0.05;	% strength of heterosynaptic LTD
tau=4;		% time-constant of neural adaptation (only used if alpha is not 0)
alpha =0;	% strength of neural adaptation

w=zeros(n); dw=zeros(n); dw2=zeros(n); dw3=zeros(n); 
x = zeros(n,1);
y=x;
xdyn=zeros(n,steps);
s=zeros(1,steps);

oldx=x;
oldy=y;

for iter=1:1000,
	
	oldw=w;

	for i = 1:steps

		b = rand(n,1)>=(1-p);

		binh = rand(n,1)>=(1-p);
		y=oldy+1/tau*(-oldy+(oldx)+binh);

		x = (w*oldx-beta*sum(oldx)+b-alpha*y)>0;
		dw = eta*(x*double(oldx)'-double(oldx)*x'); 
		dw2 = ones(n,1)*max(0, sum(w+dw,1)-m*wmax); 
		dw3 = max(0, sum(w+dw,2)-m*wmax)*ones(1,n); 
		w=min(wmax,max(0,w+dw-eta*epsilon*(dw2+dw3)-eye(n)*10000*wmax));

		oldx = x;
		oldy= y;  
		xdyn(:,i)=x;

	end

	subplot(2,2,1); 
	imagesc(w,[0,wmax]); colormap(hot); colorbar
	title('W'); xlabel('neuron index'); ylabel('neuron index');
	subplot(2,2,2); 
	imagesc(w'*w,[0,wmax^2]); colormap(hot); colorbar
	title('W^T*W'); xlabel('neuron index'); ylabel('neuron index');
	subplot(2,2,3); 
	imagesc(w-oldw); colormap(hot); colorbar
	title('change in W'); xlabel('neuron index'); ylabel('neuron index');
	subplot(2,2,4); 
	imagesc(xdyn); 
	title('neural activity')
	xlabel('time (steps)'); ylabel('neuron index');
	drawnow; 

end


%-----------------------------------------------------
% playback of learned activiy sequences: 
%-----------------------------------------------------

for iter=1:80,
	
	oldw=w;

	for i = 1:steps

		% 4 random bursts to initiate activity: 
		if (iter<5),
		b = rand(n,1)>=(1-p);
		elseif (iter>20)&(iter<25),
		b = rand(n,1)>=(1-p);
		elseif (iter>40)&(iter<45),
		b = rand(n,1)>=(1-p);
		elseif (iter>60)&(iter<65),
		b = rand(n,1)>=(1-p);
		else
		b=0*b; 
		end

		binh = rand(n,1)>=(1-p);
		y=oldy+1/tau*(-oldy+(oldx)+binh);

		x = (w*oldx-beta*sum(oldx)+b-alpha*y)>0;
		
		oldx = x;
		oldy= y;  
		xdyn(:,i)=x;

	end

	subplot(2,2,1); 
	title(['iter=',num2str(iter)])
	imagesc(w,[0,wmax]); colormap(hot); colorbar
	title('W'); xlabel('neuron index'); ylabel('neuron index');
	subplot(2,2,2); 
	imagesc(w'*w,[0,wmax^2]); colormap(hot); colorbar
	title('W^T*W'); xlabel('neuron index'); ylabel('neuron index');
	subplot(2,2,3); 
	imagesc(xdyn'*xdyn); 
	title(['neural activity, iter =', num2str(iter)]); xlabel('neuron index'); ylabel('neuron index');
	subplot(2,2,4); 
	imagesc(xdyn); 
	title('neural activity')
	xlabel(['time (steps), iter =', num2str(iter)]); ylabel('neuron index');
	drawnow; 
end


