clear all;
close all, clc;

dataset_names = {'Indian_Pines', 'KSC', 'Botswana','Salinas', 'Pavia_University'};
classifier_names = {'KNN', 'LDA','SVM'};
svm_para = {'-c 5000.000000 -g 0.500000 -m 500 -t 2 -q',...
    '-c 10000.000000 -g 16.000000 -m 500 -t 2 -q',...
    '-c 10000 -g 0.5 -m 500 -t 2 -q',...
    '-c 100 -g 16 -m 500 -t 2 -q',...
    '-c 100 -g 4 -m 500 -t 2 -q',...
    };
train_ratio = [0.1, 0.1, 0.1, 0.1, 0.1];

SuperpixelNum = [165, 186, 180, 103, 155];


ResSavePath = 'results/';

warning off;

dataidx = [1];

for dataset_id = dataidx
    Dataset = load_data(dataset_names{dataset_id});
    Dataset.svm_para = svm_para{1, dataset_id};
    Dataset.train_ratio = train_ratio(dataset_id);
    superpixel_num = SuperpixelNum(dataset_id);
    for classifier_id = 1
        disp(['Processing dataset: ',dataset_names{dataset_id}, 'classifier: ',classifier_names{classifier_id}]);
        [OA,MA,Kappa] = S4P_main(Dataset,classifier_names{classifier_id},superpixel_num);
        resFile = [ResSavePath dataset_names{dataset_id},'-',num2str(superpixel_num),'-',...
            num2str(train_ratio(dataset_id)),'-',classifier_names{classifier_id},'.mat'];    
        save(resFile, 'OA','MA','Kappa');
    end
end
