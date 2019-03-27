% simulation script as used in Figure 2 of main paper 
%
%
% Based on the matlab script sim_for_paper.m
% from Fig 2 in "Spike-Time-Dependent Plasticity and Heterosynaptic 
% Competition Organize Networks to Produce Long Scale-Free Sequences of 
% Neural Activity" Neuron 2010 by
% Ila R. Fiete, Walter Senn, Claude Z.H. Wang, and Richard H.R. Hahnloser
%
% See in particular Fig 2 and methods section "Parameters and Initial
% Conditions" subheading "Summed-Weight Limit, Binary Neurons"
%
% Tweaked by Andrew Leifer for NEU 422, Princeton 2017



%% Generate W matrix


n=50;       %number of neurons
steps=200;  % time-steps in one "iteration". each time-step is ~ 1 burst duration.


beta = .25;	% global inhibition strength
wmax=1;		% single synapse hard bound
m=1;		% Wmax = m*wmax 
pn=2;		 
p=pn/n;		% probability of external stimulation of any neuron at any time
eta=0.035;	% learning rate parameter (originally 0.125, paper suggests.025, Tank suggests .035)
epsilon=0.05;	% strength of heterosynaptic LTD
tau=4;		% time-constant of neural adaptation (only used if alpha is not 0)
alpha =0;	% strength of neural adaptation

PLOT_AS_YOU_GO=false;



%initialize variables
w=zeros(n); dw=zeros(n); dw2=zeros(n); dw3=zeros(n); 
x = zeros(n,1);
y=x;
xdyn=zeros(n,steps);
s=zeros(1,steps);

oldx=x;
oldy=y;
nIters=1000; % 1000 in original paper



for iter=1:nIters,
	
	oldw=w;

	for i = 1:steps

		b = rand(n,1)>=(1-p); % set the external stimulation

		binh = rand(n,1)>=(1-p);            % not relevant
		y=oldy+1/tau*(-oldy+(oldx)+binh); % y only has to do with neural adaptation.. 
                                          % it can be ignored for cases of
                                          % no adaptation

            % calculate neural activity, x
            x =...                          % neural activity
            ( w*oldx ...                % weights times input frm old activity
            -beta*sum(oldx)...          % global inhibition term
            +b...                       % external stimulation
            -alpha*y)...                % neural adaptation term
            >0;                     % neurons are only active when this 
                                    % sum is greater than zero
                                    
        % STDP 
		dw = eta*(x*double(oldx)'-double(oldx)*x');  %STDP
        
        %Heterosynaptic plasticity
		dw2 = ones(n,1)*max(0, sum(w+dw,1)-m*wmax); %test to see whether sum of weights in each row is too great 
		dw3 = max(0, sum(w+dw,2)-m*wmax)*ones(1,n); %test to see weather sum of weights in each column is too great
        
        %update the new weight matrix
		w=min(wmax,...                  % any weight that exceeds wmax gets set to wmax
            max(0,...                   % force any negative synapses to be zero
            w+dw ...                    % update weight matrix based on STDP
            -eta*epsilon*(dw2+dw3)...   % Heterosynaptic plasticity: penalize all those rows or colums whose sums exceeded the maximum synaptic weight
            -eye(n)*10000*wmax));       % Zero out any synapses from one neuron to itself 
                                        %   (remember that anything
                                        %   negative is set to zero)

		oldx = x;  % remember the old neural activity
		oldy= y;   % remember the old adatation state (not important when no adaption in model)
		xdyn(:,i)=x; % record a copy of the neural activity

 
        
        
    end

    
        if (iter==nIters)  || PLOT_AS_YOU_GO

            % Andy's code:

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




            subplot(2,4,1); 
            imagesc(w,[0,wmax]); colormap(hot); colorbar
            title('W'); xlabel('neuron index'); ylabel('neuron index');

            subplot(2,4,2); 
            imagesc(w'*w,[0,wmax^2]); colormap(hot); colorbar
            title('W^T*W'); xlabel('neuron index'); ylabel('neuron index');

            ax=subplot(2,4,3);
            imagesc(w(playbacksrt,playbacksrt))
            title(['w sorted  iter=' num2str(iter) ' of ' num2str(nIters)])
            %If you get an error here it is because MATLAB changed the way they
            %label axes in version 2016
            sortedticklabels=cellfun(@num2str,num2cell(playbacksrt),'UniformOutput',false);
            try
                ax.YTick=[1:n];
                ax.XTick=[1:n]; 
                ax.YTickLabel=sortedticklabels;
                ax.XTickLabel=sortedticklabels;

            catch
                yticks(1:n);
                xticks(1:n);
                yticklabels(sortedticklabels);
                xticklabels(sortedticklabels);
            end
            ylabel('neuron index')
            xlabel('neuron index')

            subplot(2,4,4);
            hist(reshape(w,1,[]));
            title('W')
            ylabel('Counts')
            xlabel('Weight')



            subplot(2,4,5); 
            imagesc(w-oldw); colormap(hot); colorbar
            title('change in W'); xlabel('neuron index'); ylabel('neuron index');


            subplot(2,4,6); 
            imagesc(xdyn); 
            title('neural activity')
            xlabel('time (steps)'); ylabel('neuron index');


            ax=subplot(2,4,7); 
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



            drawnow;
        end
        
        if mod(iter,100)==0
            disp(['Iteration ' num2str(iter)]);
        end
    

end


% Turn off plasticity and observe activity in the network
observeNetworkActivity(w)
