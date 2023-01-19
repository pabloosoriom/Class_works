data=readtable('hidden_layer_output.csv');
data(1,:) = [];
data(:,1)=[];
data=table2array(data);


Indices=randperm(length(data));
trnData=data(Indices(1:length(Indices)/2),:);
chkData=data(Indices(length(Indices)/2+1:end),:);

plot((1:length(trnData)),trnData(:,end))

%Building initial fuzzy system

fis = genfis(trnData(:,1:end-1),trnData(:,end),...
    genfisOptions('GridPartition'));
figure
subplot(4,4,1)
plotmf(fis,'input',1)
subplot(4,4,2)
plotmf(fis,'input',2)
subplot(4,4,3)
plotmf(fis,'input',3)
subplot(4,4,4)
plotmf(fis,'input',4)
subplot(4,4,5)
plotmf(fis,'input',5)
subplot(4,4,6)
plotmf(fis,'input',6)
subplot(4,4,7)
plotmf(fis,'input',7)
subplot(4,4,8)
plotmf(fis,'input',7)




%Train ANFIS model
options = anfisOptions('InitialFIS',fis,'ValidationData',chkData,'EpochNumber',15);
[fis1,error1,ss,fis2,error2] = anfis(trnData,options);

%Plotting error
figure
plot([error1 error2])
hold on
plot([error1 error2],'o')
legend('Training error','Checking error')
xlabel('Epochs')
ylabel('Root Mean Squared Error')
title('Error Curves')

%Comparasion to prediction
anfis_output = evalfis(fis2,[trnData(:,1:8); chkData(:,1:8)]);

figure
plot((1:length(data)),[data(:,end) anfis_output])
xlabel('X')
ylabel('Y')
title('Curves comparison')

%PRedictor errors
% diff = x(index) - anfis_output;
% plot(time(index),diff)
% xlabel('Time (sec)')
% title('Prediction Errors')