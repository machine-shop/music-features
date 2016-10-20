clear all
close all
clc

% Directories
% homedir = getenv('HOME');
% mirdir = fullfile(homedir,  'matlab-toolbox', 'MIRtoolbox1.6.1', 'MIRToolbox');
% datadir = fullfile(homedir, 'python-packages', 'music.features', 'src', 'tests', 'data');

% Lucine's Computer directory
datadir = fullfile('..', 'data');

% Matlab path to MIRToolbox
% addpath(genpath(mirdir));

fname = fullfile(datadir, 'Beethoven_Op031No2-03_002_20090916-SMD.wav');
fname2 = fullfile(datadir, 'test.wav');
fname3 = fullfile(datadir, 'test_1000,350,350,0,1000.wav');

a = miraudio(fname);
a2 = miraudio(fname2);
a3 = miraudio(fname3);

%cd(datadir)
%a = miraudio('Folder');

% RMS
% RMS = mirrms(a)
% RMS2 = mirrms(a2)
% RMS3 = mirrms(a3)
% RMS_frame = mirrms(a, 'Frame')
% RMS_frame2 = mirrms(a2, 'Frame')
% RMS_frame3 = mirrms(a3, 'Frame')

%Various ZeroCross tests
ZCR = mirzerocross(a)
ZCR_sample = mirzerocross(a,'Per','Sample')
ZCR_both = mirzerocross(a,'Dir','Both')
ZCR_sample_both = mirzerocross(a,'Per','Sample','Dir','Both')
ZCR_frame = mirzerocross(a, 'Frame')

ZCR2 = mirzerocross(a2)
ZCR2_sample = mirzerocross(a2,'Per','Sample')
ZCR2_both = mirzerocross(a2,'Dir','Both')
ZCR2_sample_both = mirzerocross(a2,'Per','Sample','Dir','Both')
ZCR2_frame = mirzerocross(a2, 'Frame')

ZCR3 = mirzerocross(a3)
ZCR3_sample = mirzerocross(a3,'Per','Sample')
ZCR3_both = mirzerocross(a3,'Dir','Both')
ZCR3_sample_both = mirzerocross(a3,'Per','Sample','Dir','Both')
ZCR3_frame = mirzerocross(a3, 'Frame')

% SpectralCentroid
% SC = mircentroid(a);
% SC2 = mircentroid(a2);
% SC3 = mircentroid(a3);
% SC_frame = mircentroid(a, 'Frame');
% SC2_frame = mircentroid(a2, 'Frame');
% SC3_frame = mircentroid(a3, 'Frame');

% SpectralSpread
% SS = mirspread(a)
% SS2 = mirspread(a2)
% SS3 = mirspread(a3)
% SS_frame = mirspread(a, 'Frame')
% SS2_frame = mirspread(a2, 'Frame')
% SS3_frame = mirspread(a3, 'Frame')

% Spectrum Tests
% mirspectrum(a)
% mirspectrum(a, 'Window')
% mirspectrum(a, 'Normal')
% mirspectrum(a, 'NormalLength')

% SpectralFlatness
% SF = mirflatness(a)
% SF2 = mirflatness(a2)
% SF3 = mirflatness(a3)
% SF_frame = mirflatness(a, 'frame')
% SF2_frame = mirflatness(a2, 'frame')
% SF3_frame = mirflatness(a3, 'frame')


% Key
% KEY = mirkey(a, 'Frame') 
% KEY = mirkey(a) 

% Key Strength
% mirkeystrength(a) %KStrength =

% Chromagram
% c = mirchromagram(a, 'Wrap', 'yes')

% mirfluctuation(a,'Mel','Frame', 3, 10) %,'Frame', 3, 1)
