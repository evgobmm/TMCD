import math
import torch
import scipy.special as sc
from scipy.stats import chi2
import random
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def _invSqrt(A):
        vals, vecs = torch.linalg.eigh(A)
        return vecs @ torch.diag(1.0 / torch.sqrt(vals)) @ vecs.T

@torch.no_grad()
def rTensorNorm(n, M, Sigma1, Sigma2, Sigma3):
    p1 = Sigma1.shape[0]
    p2 = Sigma2.shape[0]
    p3 = Sigma3.shape[0]
    e1_vals, e1_vecs = torch.linalg.eigh(Sigma1)
    sqrtSigma1 = e1_vecs @ torch.diag(torch.sqrt(e1_vals)) @ e1_vecs.T
    e2_vals, e2_vecs = torch.linalg.eigh(Sigma2)
    sqrtSigma2 = e2_vecs @ torch.diag(torch.sqrt(e2_vals)) @ e2_vecs.T
    e3_vals, e3_vecs = torch.linalg.eigh(Sigma3)
    sqrtSigma3 = e3_vecs @ torch.diag(torch.sqrt(e3_vals)) @ e3_vecs.T
    Z = torch.randn(n, p1, p2, p3, device=device)
    Z = Z.view(n * p1 * p2, p3) @ sqrtSigma3
    Z = Z.view(n, p1, p2, p3)
    Z = Z.permute(0, 1, 3, 2).contiguous()
    Z = Z.view(n * p1 * p3, p2) @ sqrtSigma2
    Z = Z.view(n, p1, p3, p2)
    Z = Z.permute(0, 3, 2, 1).contiguous()
    Z = Z.view(n * p2 * p3, p1) @ sqrtSigma1
    Z = Z.view(n, p2, p3, p1)
    Z = Z.permute(0, 3, 1, 2).contiguous()
    M_expanded = M.unsqueeze(0).expand(n, p1, p2, p3)
    Z = Z + M_expanded
    return Z

@torch.no_grad()
def computeTensorMD(C, invSqrt1, invSqrt2, invSqrt3, returnContributions=False):
    n, p1, p2, p3 = C.shape
    C = C.view(n * p1 * p2, p3) @ invSqrt3
    C = C.view(n, p1, p2, p3)
    C = C.permute(0, 1, 3, 2).contiguous()
    C = C.view(n * p1 * p3, p2) @ invSqrt2
    C = C.view(n, p1, p3, p2)
    C = C.permute(0, 3, 2, 1).contiguous()
    C = C.view(n * p2 * p3, p1) @ invSqrt1
    C = C.view(n, p2, p3, p1)
    C = C.permute(0, 3, 1, 2).contiguous()
    D = C * C
    D = D.view(n, -1)
    TMDsq = D.sum(dim=1)
    if returnContributions:
        D = D.view(n, p1, p2, p3)
        return TMDsq, D
    else:
        return TMDsq

@torch.no_grad()
def updateOneCov(C, invSqrt2, invSqrt3):
    n, p1, p2, p3 = C.shape
    C = C.view(n * p1 * p2, p3) @ invSqrt3
    C = C.view(n, p1, p2, p3)
    C = C.permute(0, 1, 3, 2).contiguous()
    C = C.view(n * p1 * p3, p2) @ invSqrt2
    C = C.view(n, p1, p3, p2)
    C = C.permute(1, 0, 3, 2).contiguous()
    C = C.view(p1, n * p2 * p3)
    Sigma1 = (C @ C.T) / (n * p2 * p3)
    return Sigma1





@torch.no_grad()
def flipFlopMLE(C1, C2, C3,
                Sigma1init, Sigma2init, Sigma3init,
                invSqrt2init, invSqrt3init,
                maxIter,
                tol):

    old1 = Sigma1init
    old2 = Sigma2init
    old3 = Sigma3init
    invSqrt2 = invSqrt2init
    invSqrt3 = invSqrt3init
    for it in range(maxIter):
        Sigma1 = updateOneCov(C1, invSqrt2, invSqrt3)
        invSqrt1 = _invSqrt(Sigma1)
        Sigma2 = updateOneCov(C2, invSqrt3, invSqrt1)
        invSqrt2 = _invSqrt(Sigma2)
        Sigma3 = updateOneCov(C3, invSqrt1, invSqrt2)
        d11_1 = Sigma1[0, 0]
        d11_2 = Sigma2[0, 0]
        Sigma1 = Sigma1 / d11_1
        Sigma2 = Sigma2 / d11_2
        Sigma3 = Sigma3 * (d11_1 * d11_2)
        if it == maxIter - 1:
            break
        frobDiff = torch.sum((Sigma1 - old1) ** 2) \
                 + torch.sum((Sigma2 - old2) ** 2) \
                 + torch.sum((Sigma3 - old3) ** 2)
        if frobDiff < tol:
            break
        invSqrt3 = _invSqrt(Sigma3)
        old1 = Sigma1
        old2 = Sigma2
        old3 = Sigma3
    return {
        "Sigma1": Sigma1,
        "Sigma2": Sigma2,
        "Sigma3": Sigma3,
        "invSqrt1": invSqrt1,
        "invSqrt2": invSqrt2
    }

@torch.no_grad()
def cStep(X, C, Sigma1, Sigma2, Sigma3,
          invSqrt1, invSqrt2, invSqrt3,
          alpha,
          maxIterC,
          tolC,
          maxIterFF,
          tolFF):
    def _ld(S1, S2, S3):
        return (p2 * p3) * torch.logdet(S1) \
             + (p1 * p3) * torch.logdet(S2) \
             + (p1 * p2) * torch.logdet(S3)
    n, p1, p2, p3 = X.shape
    h = int(math.floor(alpha * n))
    ldOld = _ld(Sigma1, Sigma2, Sigma3)
    for it in range(maxIterC):
        TMDsAll = computeTensorMD(C, invSqrt1, invSqrt2, invSqrt3)
        sortedIdx = torch.argsort(TMDsAll)
        subsetIndices = sortedIdx[:h]
        Xsub = X[subsetIndices]
        subMean = Xsub.mean(dim=0)
        C1 = Xsub - subMean
        C2 = C1.permute(0, 2, 3, 1).contiguous()
        C3 = C1.permute(0, 3, 1, 2).contiguous()
        initFF = flipFlopMLE(
            C1, C2, C3,
            Sigma1, Sigma2, Sigma3,
            invSqrt2, invSqrt3,
            maxIterFF,
            tolFF
        )
        Sigma1 = initFF["Sigma1"]
        Sigma2 = initFF["Sigma2"]
        Sigma3 = initFF["Sigma3"]
        invSqrt1 = initFF["invSqrt1"]
        invSqrt2 = initFF["invSqrt2"]
        ldNew = _ld(Sigma1, Sigma2, Sigma3)
        if it == maxIterC - 1 or abs(ldNew - ldOld) < tolC:
            break
        ldOld = ldNew
        C = X - subMean
        vals, vecs = torch.linalg.eigh(Sigma3)
        invSqrt3 = vecs @ torch.diag(1.0 / torch.sqrt(vals)) @ vecs.T
    return {
        "Sigma1": Sigma1,
        "Sigma2": Sigma2,
        "Sigma3": Sigma3,
        "invSqrt1": invSqrt1,
        "invSqrt2": invSqrt2,
        "subsetIndices": subsetIndices,
        "TMDsAll": TMDsAll,
        "ld": ldNew
    }

@torch.no_grad()
def tmcd(X,
         alpha,
         nSubsets,
         nBest,
         maxIterCshort,
         maxIterFFshort,
         maxIterCfull,
         maxIterFFfull,
         tolC,
         tolFF,
         beta):

    n, p1, p2, p3 = X.shape
    s = int(math.ceil(p1/(p2*p3) + p2/(p1*p3) + p3/(p1*p2))) + 2
    allSubsets = []
    for _ in range(nSubsets):
        allSubsets.append(torch.randperm(n)[:s].to(device))
    shortResults = []
    for i in range(nSubsets):
        idx = allSubsets[i]
        xSub = X[idx]
        subMean = xSub.mean(dim=0)
        C1 = xSub - subMean
        C2 = C1.permute(0, 2, 3, 1).contiguous()
        C3 = C1.permute(0, 3, 1, 2).contiguous()
        initSig1 = torch.eye(p1, device=device)
        initSig2 = torch.eye(p2, device=device)
        initSig3 = torch.eye(p3, device=device)
        initInvSqrt2 = torch.eye(p2, device=device)
        initInvSqrt3 = torch.eye(p3, device=device)
        shortMLE = flipFlopMLE(
            C1, C2, C3,
            initSig1, initSig2, initSig3,
            initInvSqrt2, initInvSqrt3,
            maxIterFFshort,
            tolFF
        )
        C = X - subMean
        curInvSqrt3 = _invSqrt(shortMLE["Sigma3"])
        shortRes = cStep(
            X, C,
            shortMLE["Sigma1"],
            shortMLE["Sigma2"],
            shortMLE["Sigma3"],
            shortMLE["invSqrt1"],
            shortMLE["invSqrt2"],
            curInvSqrt3,
            alpha,
            maxIterCshort,
            tolC,
            maxIterFFshort,
            tolFF
        )
        shortResults.append(shortRes)
    allLd = torch.tensor([res["ld"] for res in shortResults], device=device)
    rankLd = torch.argsort(allLd)
    topIdx = rankLd[:min(nBest, nSubsets)]
    fullResults = []
    for j in range(len(topIdx)):
        chosen = shortResults[topIdx[j]]
        xSub = X[chosen["subsetIndices"]]
        subMean = xSub.mean(dim=0)
        C = X - subMean
        invSqrt3 = _invSqrt(chosen["Sigma3"])
        fullRes = cStep(
            X, C,
            chosen["Sigma1"],
            chosen["Sigma2"],
            chosen["Sigma3"],
            chosen["invSqrt1"],
            chosen["invSqrt2"],
            invSqrt3,
            alpha,
            maxIterCfull,
            tolC,
            maxIterFFfull,
            tolFF
        )
        fullResults.append(fullRes)
    allLdFull = torch.tensor([r["ld"] for r in fullResults], device=device)
    bestFullIdx = torch.argmin(allLdFull)
    bestRaw = fullResults[bestFullIdx]
    dfMain = p1 * p2 * p3
    dfPlus = dfMain + 2
    chiAlpha = chi2.ppf(alpha, dfMain)
    cdfVal = chi2.cdf(chiAlpha, dfPlus)
    gammaAlpha = alpha / cdfVal
    S1 = bestRaw["Sigma1"]
    S2 = bestRaw["Sigma2"]
    S3 = bestRaw["Sigma3"] * gammaAlpha
    invSqrt2 = bestRaw["invSqrt2"]
    vals3, vecs3 = torch.linalg.eigh(S3)
    invSqrt3 = vecs3 @ torch.diag(1.0 / torch.sqrt(vals3)) @ vecs3.T
    TMDsqAll = bestRaw["TMDsAll"]
    cutoff = chi2.ppf(beta, dfMain)
    goodSet = torch.where((TMDsqAll / gammaAlpha) < cutoff)[0]
    finalGood = torch.unique(torch.cat([bestRaw["subsetIndices"], goodSet]))
    outliers = torch.tensor(list(set(range(n)) - set(finalGood.tolist())), device=device)
    Xgood = X[finalGood]
    M = Xgood.mean(dim=0)
    alphaHat = float(len(finalGood)) / n
    C1 = Xgood - M
    C2 = C1.permute(0, 2, 3, 1).contiguous()
    C3 = C1.permute(0, 3, 1, 2).contiguous()
    ffFinal = flipFlopMLE(
        C1, C2, C3,
        S1, S2, S3,
        invSqrt2, invSqrt3,
        maxIterFFfull,
        tolFF
    )
    S1 = ffFinal["Sigma1"]
    S2 = ffFinal["Sigma2"]
    S3 = ffFinal["Sigma3"]
    chiAlphaHat = chi2.ppf(alphaHat, dfMain)
    cdfValHat = chi2.cdf(chiAlphaHat, dfPlus)
    gammaAlphaHat = alphaHat / cdfValHat
    S3 = S3 * gammaAlphaHat
    return {
        "M": M,
        "Sigma1": S1,
        "Sigma2": S2,
        "Sigma3": S3,
        "outliers": outliers,
        "finalGood": finalGood
    }

