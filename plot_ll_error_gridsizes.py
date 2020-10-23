#!/usr/bin/env python3
import re, argparse, numpy as np, glob, os, subprocess, time
import matplotlib.pyplot as plt
from extractTargetFilesNonDim import epsNuFromRe
from extractTargetFilesNonDim import getAllData
from plot_spectra     import readAllSpectra
#from plot_eval_all_les import findBestHyperParams as findBestHyperParamsReference

colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']
colors = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
#colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
#colors = ['#abd9e9', '#74add1', '#4575b4', '#313695', '#006837', '#1a9850', '#66bd63', '#a6d96a', '#d9ef8b', '#fee08b', '#fdae61', '#f46d43', '#d73027', '#a50026', '#8e0152', '#c51b7d', '#de77ae', '#f1b6da']
colors = ['#6baed6','#2171b5','#08306b']
refcolors = ['#74c476','#31a354','#006d2c']
refcolors = ['#fd8d3c', '#e6550d', '#a63603']

clipBinSize = None # 15
topPercentile = 84
botPercentile = 16

def findIfGridAgent(traindir):
  if 'BlockAgents' in traindir: return False
  return True
def findActFreq(traindir):
  if 'act02' in traindir: return 2
  if 'act04' in traindir: return 4
  if 'act08' in traindir: return 8
  if 'act16' in traindir: return 16
  assert False
  return 0
def findBlockSize(traindir):
  if '2blocks' in traindir: return 16
  if '4blocks' in traindir: return 8
  if '8blocks' in traindir: return 4
  assert False
  return 0
def findBlockNum(traindir):
  if '2blocks' in traindir: return 2
  if '4blocks' in traindir: return 4
  if '8blocks' in traindir: return 8
  assert False
  return 0

def findBestHyperParams(path, re, token, gridSize, logSpectra, logEnStdev):
  bestDir, bestLL, bestStats = None, -1e9, None
  gridToken = 'BPD%03d' % gridSize
  actualNbins = gridSize//2 - 1
  binSize = clipBinSize if clipBinSize else actualNbins
  logSpectra, logEnStdev = logSpectra[:binSize], logEnStdev[:binSize]
  eps, nu = epsNuFromRe(re)
  # allow token to be actually a path
  if token[0] == '/': alldirs = glob.glob(token + '*' + ('RE%03d' % re) + '*')
  else:               alldirs = glob.glob(path + '/*' + ('RE%03d' % re) + '*')
  for dirn in alldirs:
    if     token not in dirn: continue
    if gridToken not in dirn: continue
    # if token did not produce any stats file (e.g. not applicable to dns):
    if bestDir is None: bestDir = dirn
    runData = getAllData(dirn, eps, nu, actualNbins, fSkip=1)
    avgLogSpec = np.log(runData['spectra'][:,:binSize])
    LL = - np.sum(((avgLogSpec - logSpectra) / logEnStdev) ** 2, axis=1)
    #print('RL size', avgLogSpec.shape, LL.shape)
    avgLL = np.mean(LL, axis=0)
    if avgLL > bestLL: bestDir, bestLL = dirn, avgLL
  assert bestDir is not None, "token %s - %d not found" % (token, re)
  return bestDir

def findDirectory(path, re, token, gridsize):
    gridToken, reToken = 'BPD%03d' % gridsize, 'RE%03d' % re
    if token[0] == '/': alldirs = glob.glob(token + '*')
    else:               alldirs = glob.glob(path + '/*')
    #print(path, alldirs)
    for dirn in alldirs:
        if gridToken not in dirn: continue
        if   reToken not in dirn: continue
        if     token not in dirn: continue
        return dirn
    assert False, 're-token combo not found'

def main_integral(runspaths, refspath, target, REs, tokens, gridSizes, ref):
  nRes, nTokens, nGrids = len(REs), len(tokens), len(gridSizes)
  fig, ax = plt.subplots(1, 1, figsize=[6, 4], frameon=False, squeeze=True)

  err_means = np.zeros((nTokens * nGrids, nRes))
  err_upper = np.zeros((nTokens * nGrids, nRes))
  err_lower = np.zeros((nTokens * nGrids, nRes))
  referr_means = np.zeros((len(gridSizes), nRes))
  referr_upper = np.zeros((len(gridSizes), nRes))
  referr_lower = np.zeros((len(gridSizes), nRes))

  for j, RE in enumerate(REs):
    colorid = 0
    eps, nu = epsNuFromRe(RE)
    logSpectra, logEnStdev, _, _ = readAllSpectra(target, [RE])
    logSpectra = logSpectra.reshape([logSpectra.size])
    logEnStdev = logEnStdev.reshape([logSpectra.size])

    for i, token in enumerate(tokens):
      for k, gs in enumerate(gridSizes):
        actualNbins = gs//2 - 1
        binSize = clipBinSize if clipBinSize else actualNbins
        dirn = findDirectory(runspaths[i], RE, token, gs)
        #print(dirn, token, gs)
        #findBestHyperParams(runspaths[i], RE, token, gs, logSpectra, logEnStdev)
        runData = getAllData(dirn, eps, nu, actualNbins, fSkip=1)
        M, S = logSpectra[:binSize], logEnStdev[:binSize]
        logE = np.log(runData['spectra'][:,:binSize])
        if logE.shape[0] < 2:
          err_means[i * nGrids + k, j] = np.NAN
          err_upper[i * nGrids + k, j] = np.NAN
          err_lower[i * nGrids + k, j] = np.NAN
          continue

        coef = - 0.5 / binSize
        LL = np.sum(coef * ((logE - M)/S)**2, axis=1)
        #print(i * nGrids + k, logE.shape, LL.shape)
        #err_means[i * nGrids + k, j] = np.mean(LL, axis=0)
        err_means[i * nGrids + k, j] = np.percentile(LL, 50, axis=0)
        err_upper[i * nGrids + k, j] = np.percentile(LL, topPercentile, axis=0)
        err_lower[i * nGrids + k, j] = np.percentile(LL, botPercentile, axis=0)

    for k, gridsize in enumerate(gridSizes):
      dirn = findDirectory(refspath, RE, ref, gridsize)
      #findBestHyperParamsReference(refspaths[i], RE, ref)
      #if 'BPD2' in dirn or 'BPD032' in dirn: gridsize =  32
      #if 'BPD4' in dirn or 'BPD064' in dirn: gridsize =  64
      #if 'BPD8' in dirn or 'BPD128' in dirn: gridsize = 128
      actualNbins = gridsize//2 - 1
      binSize = clipBinSize if clipBinSize else actualNbins
      runData = getAllData(dirn, eps, nu, actualNbins, fSkip=1)
      M, S = logSpectra[:binSize], logEnStdev[:binSize]
      logE = np.log(runData['spectra'][:,:binSize])
      if logE.shape[0] < 2:
        referr_means[k, j] = np.NAN
        referr_upper[k, j] = np.NAN
        referr_lower[k, j] = np.NAN
        continue
      coef = - 0.5 / binSize
      #norm, arg = np.log(2*np.pi * S**2), ((logE - M)/S)**2
      #LL = np.sum(coef * (norm + arg), axis=1)
      LL = np.sum(coef * ((logE - M)/S)**2, axis=1)
      #print(nTokens * nGrids + i, logE.shape, LL.shape)
      #err_means[nTokens * nGrids + i, j] = np.mean(LL, axis=0)
      referr_means[k, j] = np.percentile(LL, 50, axis=0)
      referr_upper[k, j] = np.percentile(LL, topPercentile, axis=0)
      referr_lower[k, j] = np.percentile(LL, botPercentile, axis=0)

  for i, gs in enumerate(gridSizes):
    ax.plot(REs, referr_means[i,:], '.-', color=refcolors[i])
    top, bot = referr_upper[i,:], referr_lower[i,:]
    ax.fill_between(REs, bot, top, facecolor=refcolors[i], alpha=.5)
  for i in reversed(range(nTokens * nGrids)):
    ax.plot(REs, err_means[i,:], '.-', color=colors[i])
    top, bot = err_upper[i,:], err_lower[i,:]
    ax.fill_between(REs, bot, top, facecolor=colors[i], alpha=.5)

  ax.set_yscale("symlog", linthreshy=1e-3)
  ax.set_ylim((-42, -0.2))
  #ax.set_ylim((-13.5, -0.29))
  #ax.set_yscale("log")
  ax.set_xlim((60, 205))
  ax.set_xscale("log")
  ax.set_xticks([60, 70, 82, 95, 111, 130, 151, 176, 205])
  ax.set_xticks([], True) # no minor ticks
  ax.set_xticklabels(
    ['60', '70', '82', '95', '111', '130', '151', '176', '205'])
  ax.set_yticks([-0.3, -1, -3, -10, -30])
  ax.set_yticks([], True) # no minor ticks
  ax.set_yticklabels(["-0.3", "-1", "-3", "-10", "-30"])
  ax.set_xlabel(r'$Re_\lambda$')
  ax.set_ylabel(r'$\widetilde{LL}(E_{LES}\,\|\,E_{DNS})$')
  ax.grid()

  fig.tight_layout()
  plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description = "Compute a target file for RL agent from DNS data.")
  ADD = parser.add_argument
  ADD('--target', default='./target_RK512_BPD128/',
    help="Path to target files directory")
  ADD('--tokens', nargs='+',
    help="Text token distinguishing each series of runs")
  ADD('--refs', help="Text token distinguishing each series of runs")
  ADD('--res', nargs='+', type=int, help="Reynolds numbers",
    default = [60, 65, 70, 76, 82, 88, 95, 103, 111, 120, 130, 140, 151, 163, 176, 190, 205])
  ADD('--labels', nargs='+', help="Plot labels to assiciate to tokens")
  ADD('--runspath', default=['./data_gridsizes/'], nargs='+', help="Plot labels to assiciate to tokens")
  ADD('--refspath', default='./data_gridsizes/', help="Plot labels to assiciate to refs")
  ADD('--gridSize', nargs='+', type=int, default=[32],
    help="Plot labels to assiciate to tokens")

  args = parser.parse_args()

  if len(args.runspath) == 1: args.runspath = args.runspath * len(args.tokens)

  if args.labels is None: args.labels = args.tokens
  assert len(args.labels) == len(args.tokens)

  #if len(args.gridSize) < len(args.labels):
  #    assert(len(args.gridSize) == 1)
  #    args.gridSize = args.gridSize * len(args.labels)

  main_integral(args.runspath, args.refspath, args.target, args.res, args.tokens, args.gridSize, args.refs)



