function baseline()
for icarpeta=1:16
% Author: Andrea Vedaldi

% Coffpyright (C) 2011-2013 Andrea Vedaldi
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

conf.calDir = 'dataset_both/' ;
conf.calDir2 = 'histogramas/' ;
conf.dataDir = 'resultados_baseline/modelos/' ;

fprintf('modelo numero %d \n',icarpeta);
carpeta = dir(conf.calDir) ;
carpeta = carpeta([carpeta.isdir]) ;
carpeta = {carpeta(3:end).name} ;
NoCarpeta=carpeta{icarpeta};
%NoCarpeta=NoCarpeta(9:end-11);
%NoCarpeta=['TODO_Train3_C16_repeticion',num2str(icarpeta)];
carpeta={carpeta{1:(icarpeta-1)},carpeta{icarpeta+1:end}};


%conf.imgDir= 'frames/x40';
conf.autoDownloadData = false ;
conf.numTrain = 23;
conf.numTest = 0 ;
conf.numClasses = 3 ;
conf.numWords = 1600 ;
conf.numSpatialX = [2 4] ;
conf.numSpatialY = [2 4] ;
conf.quantizer = 'kdtree' ;
conf.svm.C = 16 ;

conf.svm.solver = 'sdca' ;
%conf.svm.solver = 'sgd' ;
%conf.svm.solver = 'liblinear' ;

conf.svm.biasMultiplier = 1 ;
conf.phowOpts = {'Step', 3} ;
conf.clobber = false ;
conf.tinyProblem = true ;
conf.prefix = 'baseline' ;
conf.randSeed = 1 ;

if conf.tinyProblem
  conf.prefix = [NoCarpeta,'_Baseline_15abril'] ;
  conf.numClasses = 3 ;
  conf.numSpatialX = 1;
  conf.numSpatialY = 1;
  conf.numWords = 1600 ;
  conf.phowOpts = {'Verbose', 2, 'Sizes', 7, 'Step', 5} ;
end

conf.vocabPath = fullfile(conf.dataDir, [conf.prefix '-vocab.mat']) ;
conf.histPath = fullfile(conf.dataDir, [conf.prefix '-hists.mat']) ;
conf.modelPath = fullfile(conf.dataDir, [conf.prefix '-model.mat']) ;
conf.resultPath = fullfile(conf.dataDir, [conf.prefix '-result']) ;

randn('state',conf.randSeed) ;
rand('state',conf.randSeed) ;
vl_twister('state',conf.randSeed) ;



% -----------------------------------cd---------------------------------
%                                                           Setup data
% --------------------------------------------------------------------


grados={};
classes = {'grado_1','grado_2','grado_3'} ;

%%% abrir archivos csv para vector con clases %%%%

%%grado se colocan los archivos csv
% 
 for ca=1:length(carpeta)
%      
    
 grado = dir(fullfile(conf.calDir, carpeta{ca},'/atypia','/x20', '*.csv'))' ;
  grado = cellfun(@(x)fullfile([carpeta{ca},'/atypia/x20/'],x),{grado.name},'UniformOutput',false) ;
  grados = {grados{:}, grado{:}} ;
  %mageClass{end+1} = ci * ones(1,length(grado)) ;
   
 end


%%% filtrar solo los directorios con all_score

grados_aux={};atypia = {};

for fd=1:length(grados)
aux=cell2mat(grados(1,fd));
aux=aux(end-17:end-4);

testigo = strcmp(aux,'score_decision');
 if testigo
     grados_aux={grados_aux{:},grados{1,fd}};
     
 end
 


end
grados=grados_aux;

%%%%%%%%%%%% ordenar los scores y relacionarlos con una imagen

for od=1:length(grados)

atypia = {atypia{:},csvread(fullfile(conf.calDir,grados{1,od}))};

end
grados(2,:)={atypia{:}};
aux1={};
aux2={};
aux3={};
for or=1:length(grados)
 
    switch cell2mat(grados(2,or))
        case 1
            aux1={aux1{:},grados(:,or)}; 
        case 2
            aux2={aux2{:},grados(:,or)}; 
        case 3
            aux3={aux3{:},grados(:,or)};
            
        
    end
    
end

dataset=cat(2,aux1{:},aux2{:},aux3{:});

images =dataset(1,:); %%casos%%%
imageClass = cell2mat(dataset(2,:));



index1=vl_colsubset(1:length(aux1),conf.numTrain+conf.numTest);
        index2=vl_colsubset(length(aux1)+1:length(aux1)+length(aux2),conf.numTrain+conf.numTest);
        index3=vl_colsubset(length(aux1)+length(aux2)+1:length(aux1)+length(aux2)+length(aux3),conf.numTrain+conf.numTest);
        aux=randperm(conf.numTrain+conf.numTest);
        index1=index1(aux);
        index2=index2(aux);
        index3=index3(aux);
        indices_imagenes=[index1,index2,index3];
        selTrain=cat(2,index1(1:conf.numTrain),index2(1:conf.numTrain),index3(1:conf.numTrain));
        selTest=cat(2,index1(conf.numTrain+1:end),index2(conf.numTrain+1:end),index3(conf.numTrain+1:end));
        
        selTrain_svm=[1:length(index1)-conf.numTest];
        selTrain=cat(2,index1(1:conf.numTrain),index2(1:conf.numTrain),index3(1:conf.numTrain));
        selTest=cat(2,index1(conf.numTrain+1:end),index2(conf.numTrain+1:end),index3(conf.numTrain+1:end));
        y_labels=[1*ones(1,conf.numTrain+conf.numTest),2*ones(1,conf.numTrain+conf.numTest),3*ones(1,conf.numTrain+conf.numTest)]  ; 
        selTrain_svm =[1:conf.numTrain,conf.numTrain+conf.numTest+1:(conf.numTrain+conf.numTest)*2-conf.numTest,(conf.numTrain+conf.numTest)*2+1:(conf.numTrain+conf.numTest)*3-conf.numTest];
        selTest_svm = setdiff(1:length(y_labels),selTrain_svm);       
     

model.classes = {'grado_1','grado_2','grado_3'} ;
model.phowOpts = conf.phowOpts ;
model.numSpatialX = conf.numSpatialX ;
model.numSpatialY = conf.numSpatialY ;
model.quantizer = conf.quantizer ;
model.vocab = [] ;
model.w = [] ;
model.b = [] ;
model.classify = @classify ;


% --------------------------------------------------------------------
%                                           Compute spatial histograms
% --------------------------------------------------------------------

if ~exist(conf.histPath) || conf.clobber
  hists = {} ;
  histogramas=[];

  for ii=1:(length(selTrain)+length(selTest))
    sc=indices_imagenes(ii);
    direccion = dataset(1,sc);
    dir_aux= cell2mat(direccion);
    dir_carpeta= dir_aux(1:3);
    aux= cell2mat(direccion);
    imagen_sacar=aux(16:22);
    posibles=dir(fullfile(conf.calDir, dir_carpeta, '/frames/x40/' ,'*tiff'))';
    aux_names= {posibles.name};
    indexs = regexp(aux_names,[imagen_sacar,'\w']);
   testigo = 0;
   for op=1:length(indexs)
       
       if indexs{op}==1
           testigo=1+testigo;
           switch testigo
               case 1
                 im_aux1 =  [conf.calDir2,  dir_carpeta, '/',aux_names{op},'.mat' ] ;
               case 2
                  im_aux2 =  [conf.calDir2,  dir_carpeta, '/',aux_names{op},'.mat'] ;
               case 3
                   im_aux3 = [conf.calDir2,  dir_carpeta, '/',aux_names{op},'.mat' ] ;
               case 4
                   im_aux4 = [conf.calDir2,  dir_carpeta, '/',aux_names{op},'.mat'] ;

           end                
       
       end
   end
   im={im_aux1,im_aux2,im_aux3,im_aux4};




caracteristicas=[];
area_features=[];
perimetro_features=[];
ff_features=[];
RC_features=[];
area_min=[];
area_max=[];

perimetro_min=[];
perimetro_max=[];
ff_min=[];

ff_max=[];
RC_min=[];
RC_max=[];


vector_celdas=im;

for n=1:length(im)
    
load (cell2mat(vector_celdas(n)));
    
    
area_features=[area_features,area_pixels];
area_min=[area_min,minimos(1)];
area_max=[area_max,maximos(1)];

perimetro_features=[perimetro_features,perimetro_pixels];
perimetro_min=[perimetro_min,minimos(2)];
perimetro_max=[perimetro_max,maximos(2)];

ff_features=[ff_features,factor_forma];
ff_min=[ff_min,minimos(3)];
ff_max=[ff_max,maximos(3)];

RC_features=[RC_features,RC];
RC_min=[RC_min,minimos(4)];
RC_max=[RC_max,maximos(4)];
    
end

area_min=min(area_min);
area_max=max(area_max);

perimetro_min=min(perimetro_min);
perimetro_max=max(perimetro_max);

ff_min=min(ff_min);
ff_max=max(ff_max);

RC_min=min(RC_min);
RC_max=max(RC_max);


normal_area=(area_features-area_min)/(area_max-area_min);
normal_perimetro=(perimetro_features-perimetro_min)/(perimetro_max-perimetro_min);
normal_ff=(ff_features-ff_min)/(ff_max-ff_min);
normal_RC=(RC_features-RC_min)/(RC_max-RC_min);


hist_area=hist(normal_area,256);
hist_perimetro=hist(normal_perimetro,256);
hist_ff=hist(normal_ff,256);
hist_RC=hist(normal_RC,256);


%%%%%%%%%%gran duda%%%%%%%%%

  histogramas=[histogramas,[hist_area;hist_perimetro;hist_ff;hist_RC]];


  
  
 %  hist_aux={};
  % hist_sum=0;
   %    for hi=1:4
    %        hists_aux{hi} = getImageDescriptor(model,im{hi} );
     %       aux_sum=cell2mat(hists_aux(hi));
      %     hist_sum=hist_sum+aux_sum;
       %end
     % hist = hist_sum / sum(hist_sum) ;

  
%   for ii = 1:length(images)
%  % for ii = 1:length(images)
%     fprintf('Processing %s (%.2f %%)\n', images{ii}, 100 * ii / length(images)) ;
%     im = imread(fullfile(conf.calDir, images{ii})) ;
%     hists{ii} = getImageDescriptor(model, im);
%   end

  end
  
%  histogramas = cat(1, histogramas{:}) ;
  save(conf.histPath, 'histogramas') ;
else
  load(conf.histPath) ;
  end 

% --------------------------------------------------------------------
%                                                  Compute feature map
% --------------------------------------------------------------------
%calcula un kernel con una funcion chi2
psix = vl_homkermap(histogramas, 1, 'kchi2', 'gamma', .5) ;

% --------------------------------------------------------------------
%                                                            Train SVM
% --------------------------------------------------------------------

if ~exist(conf.modelPath) || conf.clobber
  switch conf.svm.solver
    case {'sgd', 'sdca'}
      lambda = 1 / (conf.svm.C *  length(selTrain_svm)) ;
      w = [] ;
      for ci = 1:length(classes)
        perm = randperm(length(selTrain_svm)) ;
        fprintf('Training model for class %s\n', classes{ci}) ;
        y = 2 * (y_labels(selTrain_svm) == ci) - 1 ;
        [w(:,ci) b(ci) info] = vl_svmtrain(psix(:, selTrain_svm(perm)), y(perm), lambda, ...
          'Solver', conf.svm.solver, ...
          'MaxNumIterations', 50/lambda, ...
          'BiasMultiplier', conf.svm.biasMultiplier, ...
          'Epsilon', 1e-3);
      end

    case 'liblinear'
      svm = train(imageClass(selTrain_svm)', ...
                  sparse(double(psix(:,selTrain_svm))),  ...
                  sprintf(' -s 3 -B %f -c %f', ...
                          conf.svm.biasMultiplier, conf.svm.C), ...
                  'col') ;
      w = svm.w(:,1:end-1)' ;
      b =  svm.w(:,end)' ;
  end

  model.b = conf.svm.biasMultiplier * b ;
  model.w = w ;

  save(conf.modelPath, 'model') ;
else
  load(conf.modelPath) ;
end

% --------------------------------------------------------------------
%                                                Test SVM and evaluate
% --------------------------------------------------------------------
scores = model.w' * psix + model.b' * ones(1,size(psix,2)) ;


% for c=21:69
%     if all(scores(:,c)<0)
%     scores(2,c)=1;
%     end
%     
% end



% for c=216:270
%     if all(scores(:,c)<0)
%     scores(3,c)=1;
%     end
%     
% end

[drop, imageEstClass] = max(scores, [], 1) ;


%%%% scores 2 %%%%

%imagenes40X=images;
% for s=1:length(selTest)
%  %busqueda de 4 imagenes%
%  aux=cell2mat(images(s));
%  r(1,s)=fins
% 
% end
%     
    

% Compute the confusion matrix
idx = sub2ind([length(classes), length(classes)], ...
              y_labels(selTest_svm), imageEstClass(selTest_svm)) ;
confus = zeros(length(classes)) ;



confus = vl_binsum(confus, ones(size(idx)), idx) ;

% Plots
% for hi=1:4
%             hists_aux{hi} = getImageDescriptor(model,im{hi} );
%             aux_sum=cell2mat(hist_aux{1});
%            hist_sum=hist_sum+aux_sum;
%        end
%       
figure(1) ; clf;
subplot(1,2,1) ;
imagesc(scores(:,[selTrain_svm selTest_svm])) ; title('Scores') ;
set(gca, 'ytick', 1:length(classes), 'yticklabel', classes) ;
subplot(1,2,2) ;
imagesc(confus) ;
title(sprintf('Confusion matrix (%.2f %% accuracy)', ...
              100 * mean(diag(confus)/conf.numTest) )) ;
print('-depsc2', [conf.resultPath '.ps']) ;
save([conf.resultPath '.mat'], 'confus', 'conf') ;


end





# baseline
# baseline
