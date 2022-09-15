clc
disp('Clear all...')
%clear all
%設定資料集路徑
parentDir = 'C:\Users\Hsieh73\Desktop\statistics\碩一下\Artificial Neural Networks\Homework2';
dataDir = 'MerchData';

% Divied into Training and Validation Data
disp('Divied into Training and Validation Data...')
allImages = imageDatastore(fullfile(parentDir,dataDir),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
%rng default
%分割訓練資料與測試資料集
[imgsTrain,imgsValidation] = splitEachLabel(allImages,0.8,'randomized');
disp(['Number of training images: ',num2str(numel(imgsTrain.Files))]);
disp(['Number of validation images: ',num2str(numel(imgsValidation.Files))]);

%把影像resize成網路模型的input size
augimdsTrain = augmentedImageDatastore([227 227],imgsTrain);
augimdsTest = augmentedImageDatastore([227 227],imgsValidation);
%%
% 宣告網路模型

layers = [
    imageInputLayer([227 227 3],"Name","imageinput")
    convolution2dLayer([3 3],32,"Name","conv_1","Padding","same")
    reluLayer("Name","relu_1")
    maxPooling2dLayer([5 5],"Name","maxpool_1","Padding","same")
    convolution2dLayer([3 3],32,"Name","conv_2","Padding","same")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([5 5],"Name","maxpool_2","Padding","same")
    fullyConnectedLayer(5,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

plot(layerGraph(layers));


% disp('alexnet setting...')
% net= alexnet


%%
%Set Training Options 設定網路訓練參數
Checkpoint=pwd;
options = trainingOptions('adam',...
    'MiniBatchSize',16,...
    'MaxEpochs',5,...
    'InitialLearnRate',1e-4,...
    'ValidationData',augimdsTest,...
    'ValidationFrequency',30,...
    'ValidationPatience',Inf,...
    'CheckpointPath',Checkpoint,...
    'Verbose',1,...
    'ExecutionEnvironment','multi-gpu',...
    'Plots','training-progress');

%rng default
%開始訓練網路，訓練完儲存
disp('Start training...')
trainednet = trainNetwork(augimdsTrain,layers,options);
save test_classification trainednet


%Evaluate  Accuracy
disp('Evaluate trained_net Accuracy...')
[YPred,probs] = classify(trainednet,augimdsTest);
accuracy = mean(YPred==imgsValidation.Labels);
display(['trained_net accuracy: ',num2str(accuracy)])
figure
plotconfusion(imgsValidation.Labels,YPred)





