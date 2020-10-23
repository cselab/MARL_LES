#!/usr/bin/env python3
import argparse, matplotlib.pyplot as plt, numpy as np, os, shutil, glob
from scipy import interpolate

colors = ['#377eb8', '#e41a1c', '#4daf4a', '#ff7f00', '#984ea3', '#ffff33', '#a65628', '#f781bf', '#999999']

RESSM = np.asarray([60, 65, 70, 76, 82, 88, 95, 103, 111, 120, 130, 140, 151, 163, 176, 190, 205])
CSSSM = [.2, .2075, .215, .22, .225, .23, .23, .23, .23, .23, .23, .23, .23, .23, .23, .23, .23]

def extract_normalize_probabilities(x, f, nBINS, SHIFT, nSKIPb, nSKIPe):
  nSAMPS = f.size // (nBINS + SHIFT)
  f = f.reshape([nSAMPS, nBINS + SHIFT])
  meanH = np.mean(f[nSKIPb:nSKIPe, SHIFT:], axis=0)
  stdH  = np.std( f[nSKIPb:nSKIPe, SHIFT:], axis=0)
  if SHIFT == 4:
        mean_sgs_viscosity_coef = np.mean(f[nSKIPb:nSKIPe, 1], axis=0)
  else: mean_sgs_viscosity_coef = 0 # check assert below! must be DNS
  integral = sum(meanH * (x[1]-x[0]))
  return mean_sgs_viscosity_coef, meanH/integral, stdH/integral

def readSGSfile(dirname, savedir = ""):
  fname = dirname + '/sgsAnalysis.raw'
  rl_fname = dirname + '/simulation_000_00000/run_00000000/sgsAnalysis.raw'
  dns_fname = dirname + '/dnsAnalysis.raw'
  if   os.path.isfile(fname)     :
    f = np.fromfile(fname,     dtype=np.float64)
    savedir = savedir + '/sgsAnalysis.raw'
  elif os.path.isfile(rl_fname)  :
    f = np.fromfile(rl_fname,  dtype=np.float64)
    savedir = savedir + '/sgsAnalysis.raw'
  elif os.path.isfile(dns_fname) :
    f = np.fromfile(dns_fname, dtype=np.float64)
    #os.makedirs(savedir, exist_ok=True)
    savedir = savedir + '/dnsAnalysis.raw'
  else : assert False, 'sgsAnalysis file not found' #return [0, 0, 0, 0] #

  # handles deprecate file saving sizes:
  if f.size == 181 : 
    nBINS, SHIFT = 90, 0
    x = (np.arange(nBINS) + 0.5)/nBINS * 0.09
    nu, meanH, stdH = f[0], f[1:nBINS+1], f[nBINS+1:]
  elif f.size % 904 == 0 : 
    nBINS, SHIFT = 900, 4
    nSAMPS = f.size // (nBINS + SHIFT)
    nSKIPb, nSKIPe = 0, nSAMPS
    x = (np.arange(nBINS) + 0.5)/nBINS * 0.09
    nu, meanH, stdH = extract_normalize_probabilities(x, f, nBINS, SHIFT, nSKIPb, nSKIPe)
  elif f.size % 200 == 0 : 
    # format that was used for DNS simulations, sgs disipation stats not stored:
    assert(os.path.isfile(dns_fname))
    nBINS, SHIFT = 200, 200
    nSAMPS = f.size // (nBINS + SHIFT)
    nSKIPb, nSKIPe = int(0.2 * nSAMPS), int(0.8 * nSAMPS)
    x = np.linspace(-0.1, 0.15, 200)
    nu, meanH, stdH = extract_normalize_probabilities(x, f, nBINS, SHIFT, nSKIPb, nSKIPe)
  else :
    nBINS, SHIFT = 90, 4
    nSAMPS = f.size // (nBINS + SHIFT)
    nSKIPb, nSKIPe = 0, nSAMPS
    x = (np.arange(nBINS) + 0.5)/nBINS * 0.09
    nu, meanH, stdH = extract_normalize_probabilities(x, f, nBINS, SHIFT, nSKIPb, nSKIPe)

  return nu, x, meanH, stdH

def plot(ax, dirname, i, savedir): #
  
  mean_sgs_viscosity_coef, CS, meanH, stdH = readSGSfile(dirname, savedir)

  '''
    #P =  np.zeros(nBINS)
    #for i in range(nSAMPS):
    #  MCS, VCS = f[i,0], f[i,2]
    #  denom = 1.0 / np.sqrt(2*np.pi*VCS) / nSAMPS
    #  P += denom * np.exp( -(x-MCS)**2 / (2*VCS) )
    #plt.plot(x, P)
  '''

  Yb, Yt = meanH-stdH, meanH+stdH
  # max/min here are out of y axis, eliminate bug with log plot at -inf
  meanH = np.maximum(meanH, 1e-1)
  Yb, Yt = np.maximum(Yb, 1e-1), np.maximum(Yt, 1e-1)
  #print( sum(meanH * (x[1]-x[0]) ) ) # should be 1 +/- floaterr
  ax.fill_between(CS[:-1], Yb[:-1], Yt[:-1], facecolor=colors[i], alpha=.5)
  line = ax.plot(CS[:-1], meanH[:-1], color=colors[i], label=dirname[-5:])
  return line

def findDirectory(path, re, token):
    if token[0] == '/': alldirs = glob.glob(token + '*' + ('RE%03d' % re) + '*')
    else:               alldirs = glob.glob(path + '/*' + ('RE%03d' % re) + '*')
    for dirn in alldirs:
        if token not in dirn: continue
        return dirn
    assert False, 're-token combo not found'

def main(runspath, REs, tokens, labels, savedirs):
    nRes = len(REs)
    axes, lines = [], []
    #sharey=True, 
    fig, axes = plt.subplots(1,nRes, figsize=[11.4, 2.2], frameon=False, squeeze=True)

    for j in range(nRes):
        RE = REs[j]
        csssm = CSSSM[np.where(RESSM == RE)[0][0]]
        axes[j].plot([csssm ** 2, csssm ** 2], [0.2, 240], color=colors[4])
        
        for i in range(len(tokens)):
            dirn = findDirectory(runspath, RE, tokens[i])
            #findBestHyperParams(runspath, RE, tokens[i])
            l = plot(axes[j], dirn, i, savedirs[i] + ('_RE%03d/' % RE)) # 
            if j == 0: lines += [l]

    axes[0].set_ylabel(r'$\mathcal{P}\left[\,C_s^2\right]$')
    #axes[1][0].invert_yaxis()
    #for j in range(1, nRes): axes[j].set_yticklabels([])
    for j in range(nRes):
      #axes[j].set_title(r'$Re_\lambda$ = %d' % REs[j])
      axes[j].set_yscale("log")
      axes[j].set_xlim([-1e-3, 0.09])
      axes[j].set_xticks([0.01, 0.04, 0.07])
      #axes[j].set_ylim([0.2, 225])
      axes[j].set_ylim([0.25, 150])
      axes[j].grid()
      axes[j].set_xlabel(r'$C_s^2$')
    #axes[0].legend(lines, labels, bbox_to_anchor=(-0.1, 2.5), borderaxespad=0)
    assert(len(lines) == len(labels))
    #axes[0].legend(lines, labels, bbox_to_anchor=(0.5, 0.5))
    #fig.subplots_adjust(right=0.17, wspace=0.2)
    #axes[-1].legend(bbox_to_anchor=(0.25, 0.25), borderaxespad=0)
    #axes[-1].legend(bbox_to_anchor=(1, 0.5),fancybox=False, shadow=False)
    #fig.legend(lines, labels, loc=7, borderaxespad=0)
    fig.tight_layout()
    #fig.subplots_adjust(right=0.75)
    plt.show()
    #axes[0].legend(loc='lower left')

if __name__ == '__main__':
  p = argparse.ArgumentParser(description = "CSS plotter.")
  p.add_argument('tokens', nargs='+', help="Text token distinguishing each series of runs")
  p.add_argument('--path', default='./data/', help="Simulation dira patb.")
  p.add_argument('--res', nargs='+', type=int, help="Reynolds numbers",
    default = [60, 82, 111, 151, 190, 205])
  #[65, 70, 76, 88, 95, 103, 120, 130, 140, 163, 176]
  p.add_argument('--labels', nargs='+', help="Plot labels to assiciate to tokens")
  p.add_argument('-s', '--savedirs', nargs='+', help="Path to target DNS files directory")
  args = p.parse_args()
  if args.labels is None: args.labels = args.tokens
  if args.savedirs is None: args.savedirs = len(args.labels) * ['']
  assert len(args.labels) == len(args.tokens)

  main(args.path, args.res, args.tokens, args.labels, args.savedirs)
