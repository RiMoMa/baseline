function [rutas_test, claseSel, scoring  ] = evaluacion_modelos_train()
%%%recordatorio: 8 capas son 16 capas y 88 capas son 8 capas
%SIN_A0AD_V4_colorextract_mser0016_3x8capas_1600words-model

for caso=1:1
    confusiones={};


for icarpeta=1:16
  %  clear all;
    ruta= 'data_test/';
    conf.calDir2 = 'histogramas/' ;
%conf.dataDir = 'resultados_baseline/modelos/' ;
    
carpeta = dir(ruta) ;
carpeta = carpeta([carpeta.isdir]) ;
carpeta = {carpeta(3:end).name} ;
NoCarpeta=carpeta{icarpeta};
carpeta={carpeta{1:(icarpeta-1)},carpeta{icarpeta+1:end}};
fprintf(' carpeta: %s num: %d de modelo: %d \n', NoCarpeta, icarpeta,caso);carpeta={NoCarpeta};
%NoCarpeta=NoCarpeta(9:end-11);
cargade=['resultados_baseline/modelos/',NoCarpeta,'_Baseline_5abril-model.mat'];
load (cargade);

rutas_test={};
proof=['6abril_baseline_',num2str(caso),'_',NoCarpeta,'_3x16capas_mser0016_evaluacion-C4.mat'];
if  ~exist(proof)

for ca=1:length(carpeta)
     
 test = dir(fullfile(ruta, carpeta{ca},'/frames/x20/', '*.tiff'))' ;
  test = cellfun(@(x)fullfile([carpeta{ca},'/frames/x20/'],x),{test.name},'UniformOutput',false) ;
  rutas_test= {rutas_test{:},test{:}};
  %mageClass{end+1} = ci * ones(1,length(grado)) ;
   
end

indices_casos=(1:length(rutas_test));

 hists = {} ;
  histogramas=[];
for ii=1:length(indices_casos)
%for ii=1:1
      
	sc=indices_casos(ii);
    direccion = rutas_test(1,sc);
    dir_aux= cell2mat(direccion);
    dir_carpeta= dir_aux(1:3);
    aux= cell2mat(direccion);
    imagen_sacar=aux(16:22);
    posibles=dir(fullfile(ruta, dir_carpeta, '/frames/x40/' ,'*tiff'))';
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
end
   [className{ii}, score{ii}] = classify(model, histogramas);
   
length(indices_casos)

claseSel=className;
scoring=score;
grabaren=proof;
save (grabaren)
else
    
    grabaren=proof;
end
paraconfundir=load(grabaren);

[conmatriz,orde,nulos]=confumatriz_challenge15(grabaren,carpeta);

%%%generador matriz confus
matrixcaso=zeros(3,3);


if length(orde)==2
   
    matrixcaso(orde(1),orde(1))=conmatriz(1);
  matrixcaso(orde(1),orde(2))=conmatriz(3);
  matrixcaso(orde(2),orde(1))=conmatriz(2);
  matrixcaso(orde(2),orde(2))=conmatriz(4);
   
elseif length(orde)==3
    matrixcaso=conmatriz;
else
matrixcaso(orde,orde)=conmatriz;
end

total=0;
%%trues 
for f=1:9
        total=matrixcaso(f)+total;
end
 trues=0;
for t=1:3
    trues=matrixcaso(t,t)+trues;
end

falses=total-trues;
confusiones={confusiones{:},{'Caso:',NoCarpeta;'Matriz',matrixcaso;'trues',trues;'falses',falses;'total',total}};




clear score scoring test className claseSel hists_aux
end
confusiones=cat(1,confusiones{:});
archivoconfus=['C4_num',num2str(caso),'_','_6deabril_baseline','_confusma']
save (archivoconfus,'confusiones')
end


function hist = getImageDescriptor(model, im)
% -------------------------------------------------------------------------

%im = standarizeImage(im) ;
width = size(im,2) ;
height = size(im,1) ;
numWords = size(model.vocab, 2) ;

% get PHOW features
%[frames, descrs] = vl_phow(im, model.phowOpts{:}) ;
[frames, descrs] = GetDescriptors(im) ;

% quantize local descriptors into visual words
switch model.quantizer
  case 'vq'
    [drop, binsa] = min(vl_alldist(model.vocab, single(descrs)), [], 1) ;
  case 'kdtree'
    binsa = double(vl_kdtreequery(model.kdtree, model.vocab, ...
                                  single(descrs), ...
                                  'MaxComparisons', 50)) ;
end

for i = 1:length(model.numSpatialX)
  binsx = vl_binsearch(linspace(1,width,model.numSpatialX(i)+1), frames(1,:)) ;
  binsy = vl_binsearch(linspace(1,height,model.numSpatialY(i)+1), frames(2,:)) ;

  % combined quantization
  bins = sub2ind([model.numSpatialY(i), model.numSpatialX(i), numWords], ...
                 binsy,binsx,binsa) ;
  hist = zeros(model.numSpatialY(i) * model.numSpatialX(i) * numWords, 1) ;
  hist = vl_binsum(hist, ones(size(bins)), bins) ;
  hists{i} = single(hist / sum(hist)) ;
end
hist = cat(1,hists{:}) ; 




% --------------barra -----------------------------------------------------------
function [f_mrdescr,c] = GetDescriptors (im)
% -------------------------------------------------------------------------

 aux=im;
   %separación H&E
    [Inorm1 H1 E1] = normalizeStaining(aux);
    
    rgb_aux=rgb2gray(H1);
    rgb_aux=medfilt2(rgb_aux);
    I=uint8(rgb_aux);
    I=impyramid(I,'reduce');
    I=impyramid(I,'expand');
    rgb_aux=I;
     %%%%%%% MSER %%%%%%% (omito el mapa binario)
     disp('Calculando MSER features...');
     [r,f]=vl_mser(rgb_aux,'Delta',5,'DarkOnBright',1,'BrightOndark',0,'MaxArea',0.0016, 'MinDiversity',0.8,'MinArea',0.0001);
    %%corte de los candidatos
       %cortar
              f=vl_ertr(f);

       
       nCandidatos=size(r,1);
    fprintf('Numero total de candidatos %d \n', nCandidatos);
    
    %%%%%gg%%%%% Cortar los parches %%%%%%%%%%

   im=rgb2gray(H1);
  %  im=single(im);
    d={};
   % parfor x=1:length(f)
   aux_sel=randperm(length(f));
   
   
   if length(aux_sel)<1600
       palabras=length(aux_sel);
   else
       palabras=1600;
   end
   
       vector1 = f(1,[aux_sel]);
       vector2 = f(2,[aux_sel]);

	fc=[vector1;vector2];
       %fc = [vector1;vector2;32*ones(size(vector1));zeros(size(vector1))];
       %[f_sift,c]=vl_sift(im,'frames',fc);  

	[f_mrdescr, c]=extractMrDescriptor2(3,16,im,fc);  
        %eliminar info en 5 8 irrelevante
  c=c(16*16+1:end,:);
    
   
   
 

   
% -------------------------------------------------------------------------
% function [f_mrdescr,c] = GetDescriptors (im)
% % -------------------------------------------------------------------------
% 
%  aux=im;
%    %separación H&E
%     [Inorm1 H1 E1] = normalizeStaining(aux);
%     
%     rgb_aux=rgb2gray(H1);
%     rgb_aux=medfilt2(rgb_aux);
%     I=uint8(rgb_aux);
%     I=impyramid(I,'reduce');
%     I=impyramid(I,'expand');
%     rgb_aux=I;
%      %%%%%%% MSER %%%%%%% (omito el mapa binario)
%      disp('Calculando MSER features...');
%      [r,f]=vl_mser(rgb_aux,'Delta',5,'DarkOnBright',1,'BrightOndark',0,'MaxArea',0.001, 'MinDiversity',0.8,'MinArea',0.0001);
%  
%      f=vl_ertr(f);
%      %%corte de los candidatos
%        %cortar
%        nCandidatos=size(r,1);
%     fprintf('Numero total de candidatos %d \n', nCandidatos);
%     
%     %%%%%gg%%%%% Cortar los parches %%%%%%%%%%
% 
%    im=rgb2gray(im);
%    %im=single(im);
%     d={};
%    % parfor x=1:length(f)
%    aux_sel=randperm(length(f));
%    
%    
%    if length(aux_sel)<800
%        palabras=length(aux_sel);
%    else
%        palabras=800;
%    end
%    
%        vector1 = f(1,[aux_sel]);
%        vector2 = f(2,[aux_sel]);
% 
% 	fc=[vector1;vector2];
%        %fc = [vector1;vector2;32*ones(size(vector1));zeros(size(vector1))];
%        %[f_sift,c]=vl_sift(im,'frames',fc);  
% 
% 	[f_mrdescr, c]=extractMrDescriptor(3,8,im,fc,false);  
%  

   





%%%%%%%%recibe una imagen y la clasifica %%%%%%%%%

% -------------------------------------------------------------------------
 function [className, score] = classify(model, hist)
% % -------------------------------------------------------------------------
psix = vl_homkermap(hist, 1, 'kchi2', 'gamma', .5) ;
scores = model.w' * psix + model.b' ;
[score, best] = max(scores) ;
className = model.classes{best} ;

