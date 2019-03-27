%NEU422_HW2_analysis
%% sequence model
N = 50;
A = eye(N);
A = A(randperm(N),:);
imagesc(A)

observeNetworkActivity(A);

sigma = 0.3;
AA = A+randn(N,N)*sigma;
AA(AA>1) = 1;
AA(AA<0) = 0;
observeNetworkActivity(A);

learnWeights;

%% Neural dynamics in data
load('/Users/Macintosh/Desktop/fiete_sim-master/k55_20160613_RSM.mat')

%% Behavioral correlation
window = 110;
acs = 20;
lefturn = [];
righturn = [];
for tr = 1:size(trials,1)
    thistrial = trials(tr);
    if thistrial.goodQuality==1 && thistrial.mainTrial==1
       pos = thistrial.fArmEntry;
       %at the arm entry point, we can check the position or velocity, and
       %sort with choice or types of trials
       if thistrial.choice=='L' %thistrial.trialType=='L' %
           lefturn = [lefturn; data.velocity(pos-window:pos+acs)];
           %lefturn = [lefturn; data.position(pos-window:pos+acs)];
       end
       if thistrial.choice=='R' %thistrial.trialType=='R' %
           righturn = [righturn; data.velocity(pos-window:pos+acs)];
           %righturn = [righturn; data.position(pos-window:pos+acs)];
       end
    end
end
dt = mean(diff(data.time));
plot(-window*dt:dt:acs*dt,lefturn','r')
hold on
plot(-window*dt:dt:acs*dt,righturn','b')
xlabel('time to arm entry')
ylabel('velocity')

%% Task/choice-modulated transients
window = 110;
acs = 20;
temp = [];
index = 0;
for tr = 1:size(trials,1)
    thistrial = trials(tr);
    if thistrial.goodQuality==1 && thistrial.mainTrial==1
        if thistrial.correctChoice==1  %choice-specific
            pos = thistrial.fCueEntry; %task structure
            temp = [temp; data.DFF(pos-window:pos+acs,:)'];
            index = index+1;
        end
    end
end
%%%temp is trials*neurons X time
temp = reshape(temp, index,size(temp,1)/index,size(temp,2));
%%%now Raster is tiral X neurons X time
Raster = squeeze(nanmean(temp,1));
dt = mean(diff(data.time));
imagesc(-window*dt:dt:acs*dt,1:size(Raster,1),Raster)
xlabel('time to arm entry')
ylabel('cell ID')

%% find specific cells
prefcell = [];
criteria = 5;
for nn = 1:size(data.DFF,2)
    tempL = [];  tempR = [];
    for tr = 1:size(trials,1)
        thistrial = trials(tr);
        if thistrial.goodQuality==1 && thistrial.mainTrial==1
            pos = thistrial.fMemEntry;
            if thistrial.choice=='L'; tempL = [tempL; nanmean(data.DFF(pos-window:pos+acs,nn))]; end;
            if thistrial.choice=='R'; tempR = [tempR; nanmean(data.DFF(pos-window:pos+acs,nn))]; end;  
        end
    end
    %if nanmean(tempL)/nanvar(tempL) > criteria*nanmean(tempR)/nanvar(tempR)
    %    prefcell = [prefcell nn];
    %end
    prefcell = [prefcell nanmean(tempL)/(nanmean(tempL)+nanmean(tempR))];
end
%% plotting
cellid = 198; %260
trialss= [];
for tr = 1:size(trials,1)
    thistrial = trials(tr);
    if thistrial.goodQuality==1 && thistrial.mainTrial==1
        %if thistrial.correctChoice==1  %choice-specific
        if thistrial.choice=='L'
            pos = thistrial.fMemEntry; %task structure
            trialss = [trialss; data.DFF(pos-window:pos+acs,cellid)'];
        end
    end
end
imagesc(-window*dt:dt:acs*dt,1:size(trialss,1),trialss)
xlabel('trials')
ylabel('dF/F')

figure();
plot(-window*dt:dt:acs*dt,nanmean(trialss,1))
xlabel('trials')
ylabel('dF/F')
%% Neural sequences
selected = find(prefcell>0.5);
[a,b] = max(Raster(selected,:)');
[aa,bb] = sort(b);
imagesc(Raster(selected(bb),:))
%%%z-scoring
temp = Raster(selected(bb),:);
temp = temp./var(temp');
imagesc(temp)

%% ...lower-dimension structure


