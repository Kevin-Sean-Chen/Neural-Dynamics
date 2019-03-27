function observeNetworkActivity(w)
% Accepts a weight matrix w and stimualtes the network occasionally and
% then simulates neural activity in the network. Note there is no synaptic
% plasticity in this simulation. The neural weights are static.
%
% This is adapted form the second half of the matlab script sim_for_paper.m
% from Fig 2 in "Spike-Time-Dependent Plasticity and Heterosynaptic 
% Competition Organize Networks to Produce Long Scale-Free Sequences of 
% Neural Activity" Neuron 2010 by
% Ila R. Fiete,1,* Walter Senn,2 Claude Z.H. Wang,3 and Richard H.R. Hahnloser3

%%
%-----------------------------------------------------
% playback of learned activiy sequences: 
%-----------------------------------------------------
figure;
Niter=80; %Ila used 80
n=length(w);
oldx=zeros(n,1);
oldy=zeros(n,1);

steps=200;  % time-steps in one "iteration". each time-step is ~ 1 burst duration.
beta = .25;	% global inhibition strength
wmax=1;		% single synapse hard bound
m=1;		% Wmax = m*wmax 
pn=2;		 
p=pn/n;		% probability of external stimulation of any neuron at any time
tau=4;		% time-constant of neural adaptation (only used if alpha is not 0)
alpha =0;	% strength of neural adaptation





            %Now, using ONLY the weight matrix, generate an ordering to display
            %the neurons so that they also show sequences
            bw=w>.5; %binarize w
            playbacksrt=1;

            while ~isempty(setdiff(1:n,playbacksrt))
                loc=find(bw(playbacksrt(end),:)==1); %find the first "1" element in W and store its location
                if ~isempty(loc) % if we found something
                    if ~ismember(loc(1),playbacksrt) %if we haven't already found that neuron
                        playbacksrt(end+1)=loc(1); % store it and move on
                    else
                        %the chain is broken and the next neuron should be the first neuron that you
                        %haven't looked at yet
                        remainingNeurons=setdiff(1:n,playbacksrt);
                        playbacksrt(end+1)=remainingNeurons(1);
                    end


                else
                        %the chain is broken and the next neuron should be the first neuron that you
                        %haven't looked at yet
                        remainingNeurons=setdiff(1:n,playbacksrt);
                        playbacksrt(end+1)=remainingNeurons(1);
                end
            end



for iter=1:Niter, 
	
	oldw=w; % not relevant

	for i = 1:steps

		% 4 random bursts to initiate activity: 
		if (iter<5),
            b = rand(n,1)>=(1-p);
            DRIVENTEXT='Driving Activity ';
		elseif (iter>20)&(iter<25),
            b = rand(n,1)>=(1-p);
            DRIVENTEXT='Driving Activity ';
		elseif (iter>40)&(iter<45),
            b = rand(n,1)>=(1-p);
            DRIVENTEXT='Driving Activity ';
		elseif (iter>60)&(iter<65),
            b = rand(n,1)>=(1-p);
            DRIVENTEXT='Driving Activity ';
        else
            b=0*b; 
            DRIVENTEXT='';
        end
        
        
        binh = rand(n,1)>=(1-p); % not relevant (adaptation)
		y=oldy+1/tau*(-oldy+(oldx)+binh); % not relevant (adaptation)

        %New neural activity
		x = (w*oldx... %weights times old neural activity
            -beta*sum(oldx)... %minus global inhibition
            +b... %plus random external activity
            -alpha*y)...%ignore the alpha
            >0; %force activity to be positive
		

        
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
    imagesc(xdyn);
    
    
	title([DRIVENTEXT 'neural activity, iter =', num2str(iter)]); xlabel('neuron index'); ylabel('neuron index');
    ax=subplot(2,2,4); 
    imagesc(xdyn(playbacksrt',:));
            %If you get an error here it is because MATLAB changed the way they
            %label axes in version 2016
            sortedticklabels=cellfun(@num2str,num2cell(playbacksrt),'UniformOutput',false);
            try
                ax.YTick=[1:n];
                ax.YTickLabel=sortedticklabels;
            catch
                yticks(1:n);
                yticklabels(sortedticklabels);
            end            
            title('neural activity')
            xlabel('time (steps)'); ylabel('neuron');
    
	title([DRIVENTEXT 'neural activity'])
	xlabel(['time (steps), iter =', num2str(iter)]); ylabel('neuron index (SORTED)');
	drawnow; 
end
