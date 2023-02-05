import datetime
import scipy.stats
from operator import itemgetter

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def correlation(QPP, MRR):
    predictedQPP = []
    ActualMRR = []
    for query, QPPScore in QPP.items():
        predictedQPP.append(float(round(QPPScore,4)))
        ActualMRR.append(float(MRR[query]))
    pearsonr, pearsonp = scipy.stats.pearsonr(ActualMRR, predictedQPP)
    kendalltau = scipy.stats.kendalltau(ActualMRR, predictedQPP)
    spearmanr = scipy.stats.spearmanr(ActualMRR, predictedQPP)

    return pearsonr, pearsonp, kendalltau.correlation, kendalltau.pvalue, spearmanr.correlation, spearmanr.pvalue

def save(config, Run):
    runFileName = config['outputPath'] +  config['outputName'] + 'Run.txt'


    runFile = open(runFileName, 'w')
    for query, retrievalList in Run.items():
        rank = 1
        retrievalList = sorted(retrievalList, key=itemgetter(1) ,reverse=True)
        for doc, retrievalScore in retrievalList:
            runFile.write(str(query) + '\t' + 'Q0' + '\t' + str(doc) + '\t' + str(rank) + '\t'+ str(retrievalScore) + '\t' + 'M-QPPF\n')
            rank = rank + 1

    runFile.close()

