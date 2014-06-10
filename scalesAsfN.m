function scalesAsfN()
% Wrapper for the nw model 

% Save library paths
   MatlabPath = getenv('LD_LIBRARY_PATH');
   Matpath = getenv('PATH');
   % Make Matlab use system libraries
   setenv('LD_LIBRARY_PATH','/usr/local/cuda-6.0/lib64:');
   setenv('PATH', ['/usr/local/cuda-6.0/bin:/usr/lib/lightdm/lightdm:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games']);
   %   [fldr, contrast, muE, muI, inputTStart, inputTStop, inputStepSize] ...
   %     = DefaultArgs(varargin, {'~/Documents/cnrs/code/network_model/', [0.25 : 0.25 : 1], 0.1, 0.1, 0, 0, 0});
   fldr = '/home/shrisha/Documents/cuda/exp/ode/';
   
   N = 10:20:200;
   for n = 1 : length(N)
       system(['sed -i ''s_#define K .*_#define K ', num2str(N(n)), '.0_'' ', 'devHostConstants.h']);
       system('make clean')
       system('make');
       system('./a.out');
       vm = importdata('vm.csv');
       st = importdata('spkTimes.csv');
       i = vm(:, [2:2:end]);
       meanI(:, n) = mean(i, 2);
       nSpks(n) = size(st, 1);
       %       elapsedTime = 
       %       system(['mv ', fldr, 'isynapEI ', fldr, 'isynapEI_K', num2str(K(ik)), '_N', num2str(N)]);
       %       system(['mv ', fldr, 'spkTimes ', fldr, 'spkTimes_K', num2str(K(ik)), '_N', num2str(N)]);
   end
   keyboard;
end