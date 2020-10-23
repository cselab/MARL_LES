#!/usr/bin/env python3
import re, argparse, numpy as np, glob, os, subprocess, time
import matplotlib.pyplot as plt
from extractTargetFilesNonDim import epsNuFromRe
from extractTargetFilesNonDim import getAllData
from plot_spectra     import readAllSpectra

#used in figure
colors = ['#984ea3', '#4daf4a',  '#e41a1c', '#377eb8', '#e6ab02']
#more colors
#colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']

lastCompiledBlockSize = 16
kXaxis = True

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

def findDirectory(path, re, token, gs):
    if token[0] == '/': alldirs = glob.glob(token + '*' + ('RE%03d' % re) + '*')
    else:               alldirs = glob.glob(path + '/*' + ('RE%03d' % re) + '*')
    for dirn in alldirs:
        if token not in dirn: continue
        return dirn
    assert False, 're-token combo not found'

def main_integral(runspaths, target, REs, tokens, labels, gridSize):
  nBins = [0] * len(gridSize)
  scores = [0] * len(tokens)
  for i,gs in enumerate(gridSize): nBins[i] = gs//2 - 1
  minNbins = min(nBins)
  nRes = len(REs)
  axes, lines = [], []
  if kXaxis:
    fig, axes = plt.subplots(2,nRes, sharex=True, figsize=[12, 4], frameon=False, squeeze=True)
  else:
    fig, axes = plt.subplots(2,nRes, sharey='row', figsize=[12, 4.8], frameon=False, squeeze=True)
  maxLL, minLL = [1] * len(REs), [-1] * len(REs)

  for j, RE in enumerate(REs):
    eps, nu = epsNuFromRe(RE)
    Ekscal = np.power(nu**5 * eps, 0.25)
    # read target file
    logSpectra, logEnStdev, _, _ = readAllSpectra(target, [RE])
    logSpectra = logSpectra.reshape([logSpectra.size])
    logEnStdev = logEnStdev.reshape([logSpectra.size])
    minNbins = min(minNbins, logSpectra.size)
    modes = np.arange(1, minNbins+1, dtype=np.float64) # assumes box is 2 pi
    LLTop = np.exp(logSpectra + logEnStdev)/Ekscal
    LLBot = np.exp(logSpectra - logEnStdev)/Ekscal

    xTheory = np.linspace(4, 14)
    coef = np.exp(logSpectra[1])/Ekscal * np.power(4, 5.0/3.0)
    yTheory = coef * np.power(xTheory, -5.0/3.0)

    if kXaxis:
      axes[0][j].plot(xTheory, yTheory, 'k--')
      axes[0][j].plot(modes, np.exp(logSpectra)/Ekscal, color='k')
      axes[0][j].fill_between(modes, LLBot, LLTop, facecolor='k', alpha=.5)
      axes[1][j].plot(modes, np.zeros(minNbins), 'k-')
    else:
      axes[1][j].plot(np.zeros(minNbins), modes, 'k-')
      axes[0][j].plot(yTheory, xTheory, 'k--')
      axes[0][j].plot(np.exp(logSpectra)/Ekscal, modes, color='k')
      axes[0][j].fill_betweenx(modes, LLBot, LLTop, facecolor='k', alpha=.5)

    for i, token in enumerate(tokens):
      runspath = runspaths[i]
      #dirn = findBestHyperParams(runspath, RE, tokens[i], logSpectra, logEnStdev)
      dirn = findDirectory(runspath, RE, token, gridSize[i])
      #print(dirn)
      runData = getAllData(dirn, eps, nu, nBins[i], fSkip=1)
      logE = np.log(runData['spectra'])
      #print(logE.shape[0])
      avgLogSpec, stdLogSpec = np.mean(logE, axis=0), np.std(logE, axis=0)
      assert(avgLogSpec.size == nBins[i])
      avgLogSpec = avgLogSpec[:minNbins]
      stdLogSpec = stdLogSpec[:minNbins]
      #print(avgLogSpec.shape, logSpectra.shape, logEnStdev.shape)

      LL    = (avgLogSpec              - logSpectra) / logEnStdev
      LLTop = (avgLogSpec + stdLogSpec - logSpectra) / logEnStdev
      LLBot = (avgLogSpec - stdLogSpec - logSpectra) / logEnStdev
      Ek    = np.exp(avgLogSpec) / Ekscal
      EkTop = np.exp(avgLogSpec+stdLogSpec) / Ekscal
      EkBot = np.exp(avgLogSpec-stdLogSpec) / Ekscal
      if LLTop.max() < 100: maxLL[j] = np.maximum(LLTop.max(), maxLL[j])
      if LLBot.min() >-100: minLL[j] = np.minimum(LLBot.min(), minLL[j])

      if kXaxis:
        p = axes[1][j].plot(modes, LL, label=labels[i], color=colors[i]) # , label=labels[i]
        axes[1][j].fill_between(modes, LLBot, LLTop, facecolor=colors[i], alpha=.5)
        axes[0][j].plot(modes, Ek, label=labels[i], color=colors[i]) # , label=labels[i]
        axes[0][j].fill_between(modes, EkBot, EkTop, facecolor=colors[i], alpha=.5)
      else:
        p = axes[1][j].plot(LL, modes, label=labels[i], color=colors[i]) # , label=labels[i]
        axes[1][j].fill_betweenx(modes, LLBot, LLTop, facecolor=colors[i], alpha=.5)
        axes[0][j].plot(Ek, modes, label=labels[i], color=colors[i]) # , label=labels[i]
        axes[0][j].fill_betweenx(modes, EkBot, EkTop, facecolor=colors[i], alpha=.5)

      scores[i] += np.sum(LL)
      #LLt = (0.5 * (logE - logSpectra) / logEnStdev ) ** 2
      #sumLLt = np.sqrt(np.sum(LLt, axis=1))
      #nSamples = sumLLt.size
      #print('found %d samples' % nSamples)
      #p = axes[j].plot(np.arange(nSamples), sumLLt, label=labels[i], color=colors[i])
      
      if j == 0: lines += [p]

  for j in range(nRes):
      axes[0][j].set_title(r'$Re_\lambda$ = %d' % REs[j])
      axes[1][j].grid()
      axes[0][j].grid()

  for i, token in enumerate(tokens): print(token, scores[i])

  if kXaxis:
    axes[0][0].set_ylabel(r'$E_{LES} \,/\, \eta u^2_\eta$')
    axes[1][0].set_ylabel(r'$\frac{\log E_{LES} - \mu(\log E_{DNS})}{\sigma(\log E_{DNS})}$')
    #axes[0][0].set_xscale("log")
    for j in range(nRes):
        axes[1][j].set_xlabel(r'$k \cdot L / 2 \pi$')
        axes[0][j].set_yscale("log")
        axes[0][j].set_xlim([1, 15])
        axes[1][j].set_ylim([minLL[j], maxLL[j]])
  else:
    axes[1][0].set_ylim([1, 15])
    axes[1][0].set_ylabel(r'$k \cdot L / 2 \pi$')
    axes[1][0].invert_yaxis()
    for j in range(nRes):
        #axes[1][j].set_xlabel(r'$\frac{\log E^{LES}(k) - \mu[\log E^{DNS}(k)]}{\sigma[\log E^{DNS}(k)]}$')
        axes[1][j].set_xlabel(r'$\frac{\log E_{LES} - \mu(\log E_{DNS})}{\sigma(\log E_{DNS})}$')
    axes[0][0].set_ylim([1, 15])
    axes[0][0].set_ylabel(r'$k \cdot L / 2 \pi$')
    axes[0][0].invert_yaxis()
    for j in range(nRes):
        axes[0][j].set_xlabel(r'$E_{LES} \,/\, \eta u^2_\eta$')
        axes[0][j].set_xscale("log")

  #axes[1][0].invert_yaxis()
  #for j in range(1,nRes): axes[j].set_yticklabels([])
  #axes[0].legend(lines, labels, bbox_to_anchor=(-0.1, 2.5), borderaxespad=0)
  assert(len(lines) == len(labels))
  #axes[0].legend(lines, labels, bbox_to_anchor=(0.5, 0.5))
  #fig.subplots_adjust(right=0.17, wspace=0.2)
  #axes[0][-1].legend(bbox_to_anchor=(0.25, 0.25), borderaxespad=0)
  #axes[-1].legend(bbox_to_anchor=(1, 0.5),fancybox=False, shadow=False)
  #fig.legend(lines, labels, loc=7, borderaxespad=0)
  fig.tight_layout()
  #fig.subplots_adjust(right=0.75)
  plt.show()
  #axes[0].legend(loc='lower left')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description = "Compute a target file for RL agent from DNS data.")
  ADD = parser.add_argument
  ADD('tokens', nargs='+',
    help="Directory naming tokens shared by all runs across Re")
  ADD('-t', '--target', help="Path to target DNS files directory",
      default='./target_RK512_BPD032/')
  ADD('--res', nargs='+', type=int, default = [60, 82, 111, 151, 190, 205],
      help="Reynolds numbers to visualize")
  #[65, 70, 76, 88, 95, 103, 120, 130, 140, 163, 176],
  # default = [60, 70, 111, 151, 176, 190, 205], 
  # default = [60, 82, 111, 151, 190, 205],
  # default = [65, 76, 88, 103, 120, 140, 163],
  ADD('--labels', nargs='+', help="Plot labels to assiciate to tokens")
  ADD('-r', '--runspath', default=['./data/'], nargs='+',
    help="Relative path to evaluation runs")
  ADD('--gridSize', nargs='+', type=int, default=[32],
    help="1D grid size used by the evaluation runs")

  args = parser.parse_args()
  if args.labels is None: args.labels = args.tokens
  if len(args.runspath) == 1: args.runspath = args.runspath * len(args.tokens)
  if len(args.gridSize) < len(args.labels):
      assert(len(args.gridSize) == 1)
      args.gridSize = args.gridSize * len(args.labels)
  assert len(args.labels) == len(args.tokens)

  main_integral(args.runspath, args.target, args.res, args.tokens, args.labels, args.gridSize)



