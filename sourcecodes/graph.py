#Scroll to bottom, uncomment those codes to run.

# -*- coding: utf-8 -*-
from __future__ import division
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import random as random
import csv
import numpy as np
import math
import powerlaw
import sys

print("\n\n\nWARNING:Program is tested only on python version 2.7, your version is "+ str(sys.version)+"\n\n")


def saveGraphAsCSV(graph, tocsvfile):
    """
    Save NetworkX graph object into an edge file (CSV).
    NetworkX save it as a file with spaces, which cannot be properly processed by Excel for further manipulation.
    Thus will convert it into proper CSV myself.
    :param graph:
    :param tocsvfile:
    :return:
    """
    cleansedfilewithspace = 'temp'+ str(random.random()) +'.edgelist'
    print "Size of this subgraph: ", len(graph.nodes()), " First node is: ", graph.nodes()[0]
    nx.write_edgelist(graph, cleansedfilewithspace, data=True)
    edgelist=[]
    with open(cleansedfilewithspace, 'r') as csvfile:
        countryreader = csv.reader(csvfile, delimiter=' ')
        for row in countryreader:
            src=row[0]
            tgt=row[1]
            edgelist.append([src,tgt])

    with open(tocsvfile, 'wb') as csvfile:
        edgewriter = csv.writer(csvfile, delimiter=',')
        for item in edgelist:
            edgewriter.writerow([item[0], item[1]])


def saveWeightedGraphAsCSV(graph, tocsvfile):
    """
    Save NetworkX weighted graph object into an edge file (CSV).
    NetworkX save it as a file with spaces, which cannot be properly processed by Excel for further manipulation.
    Thus will convert it into proper CSV myself.
    :param graph:
    :param tocsvfile:
    :return:
    """
    cleansedfilewithspace = 'temp'+ str(random.random()) +'.edgelist'
    print "Size of this subgraph: ", len(graph.nodes()), " First node is: ", graph.nodes()[0]
    nx.write_weighted_edgelist(graph, path=cleansedfilewithspace)
    edgelist=[]
    with open(cleansedfilewithspace, 'r') as csvfile:
        countryreader = csv.reader(csvfile, delimiter=' ')
        for row in countryreader:
            src=row[0]
            tgt=row[1]
            weight=row[2]
            edgelist.append([src,tgt,weight])

    with open(tocsvfile, 'wb') as csvfile:
        edgewriter = csv.writer(csvfile, delimiter=',')
        for item in edgelist:
            edgewriter.writerow([item[0], item[1], item[2]])

def cleanseEdgeData(csvfile,cleansedcsvfile, ctyarray):
    """
    There are errors in data pertaining to UN country nodes. These nodes may not exist either due to spelling or they refer to regional entities.
    Such nodes are detected and removed
    :param edgefile:
    :param ctyarray: A list of UN Countries
    :return:
    """
    cleansedfilewithspace = 'temp'+ str(random.random()) +'.edgelist'
    graph = nx.read_edgelist(csvfile, delimiter=',', create_using=nx.DiGraph())
    for n in graph.nodes (): #For each node in the graph
        try:
            ctylongform=ctyarray[n]
        except:
            print "This node doesn't exists in country list...dropping node ", n
            graph.remove_node(n)
    nx.write_edgelist(graph, cleansedfilewithspace,data=False)

    edgelist=[]
    with open(cleansedfilewithspace, 'r') as csvfile:
        countryreader = csv.reader(csvfile, delimiter=' ')
        for row in countryreader:
            src=row[0]
            tgt=row[1]
            edgelist.append([src,tgt])

    with open(cleansedcsvfile, 'wb') as csvfile:
        edgewriter = csv.writer(csvfile, delimiter=',')
        for item in edgelist:
            edgewriter.writerow([item[0], item[1]])


def cleanseWeightedEdgeData(csvfile,cleansedcsvfile, ctyarray):
    """
    There are errors in data pertaining to UN country nodes. These nodes may not exist either due to spelling or they refer to regional entities.
    Such nodes are detected and removed
    :param edgefile:
    :param ctyarray: A list of UN Countries
    :return:
    """
    cleansedfilewithspace = 'temp'+ str(random.random()) +'.edgelist'
    graph = nx.read_weighted_edgelist(csvfile, delimiter=',', create_using=nx.DiGraph())
    for n in graph.nodes (): #For each node in the graph
        try:
            ctylongform=ctyarray[n]
        except:
            print "This node doesn't exists in country list...dropping node ", n
            graph.remove_node(n)
    nx.write_weighted_edgelist(graph, cleansedfilewithspace)

    edgelist=[]
    with open(cleansedfilewithspace, 'r') as csvfile:
        countryreader = csv.reader(csvfile, delimiter=' ')
        for row in countryreader:
            src=row[0]
            tgt=row[1]
            weight=row[2]
            edgelist.append([src,tgt,weight])

    with open(cleansedcsvfile, 'wb') as csvfile:
        edgewriter = csv.writer(csvfile, delimiter=',')
        for item in edgelist:
            edgewriter.writerow([item[0], item[1],item[2]])


def writeUNCountryList(ctyarray):
    """
    Properly output the UN Country list in the form of ShortForm, LongForm.
    :param ctyarray: A list of UN Countries
    :return:
    """
    with open('country.txt', 'wb') as csvfile:
        ctywriter = csv.writer(csvfile, delimiter=',')
        items = ctyarray.items()
        for item in items:
            ctywriter.writerow([item[0], item[1]])
            # print item[0], item[1]




def loadUNCountryList(ctyfile):
    """
    Load the UN country list into a proper array
    :param ctyfile: From the raw data file UNCountryCode.txt
    :return: An array of UN Countries
    """
    ctyarray = {}
    with open(ctyfile, 'r') as csvfile:
        countryreader = csv.reader(csvfile, delimiter=' ')
        for row in countryreader:
            fullword=''
            for word in row[:-1]:
                fullword=fullword+" "+str(word)
            ctyarray[str(row[-1])]=fullword
        print "Total no of UN countries: ", len(ctyarray)
        print ctyarray
        print "ShortForm:SGP", "LongForm:", ctyarray['GBR']
        return ctyarray





def plot_degree_distribution_bined (graph, loglog=False, edgetype='undirected', binsize=10, normed=False, overlayfit=False) :
    """
    Plot the degree distribution of graph.
    :param graph:
    :param loglog: log log scale. (Used to access if power law exists)
    :param edgetype:undirected, in links only, out links only.
    :param binsize:
    :param normed: Outcome in terms of probability instead of counts.
    :return:
    """
    print "loglog=",loglog, "edgetype=",edgetype,"binsize=",binsize,"normalised=",normed
    totalNodes=float(len(graph.nodes()))
    print "Total no of countries in graph: ", totalNodes
    Nklist=[]
    degs = {} #K, Nk
    ctyOutDegree=defaultdict(str)
    ctyInDegree=defaultdict(str)
    ctyDegree = defaultdict(str)
    for n in graph.nodes () :
        deg=0
        if(edgetype=='undirected'):
            deg = graph.degree (n)
            Nklist.append(graph.degree (n))
            ctyDegree[deg] = str(ctyDegree[deg]) + "," + str(n)
        elif(edgetype=='in'):
            deg = graph.in_degree(n)
            Nklist.append(deg)
            # print "Country:", n, "  In-Degree:", deg
            ctyInDegree[deg]=str(ctyInDegree[deg])+","+str(n)
        elif(edgetype=='out'):
            deg = graph.out_degree(n)
            Nklist.append(graph.out_degree(n))
            # print "Country:", n, "  Out-Degree:", deg
            ctyOutDegree[deg] = str(ctyOutDegree[deg]) + "," + str(n)
        if deg not in degs:
            degs[deg] = 0
        degs[deg] += 1



    items = sorted(degs.items())

    if(edgetype.lower()=='out'):
        print "\nTop five busybodies:"
        print "Country:", ctyOutDegree[items[-1][0]], " ", items[-1][0]
        print "Country:", ctyOutDegree[items[-2][0]], " ", items[-2][0]
        print "Country:", ctyOutDegree[items[-3][0]], " ", items[-3][0]
        print "Country:", ctyOutDegree[items[-4][0]], " ", items[-4][0]
        print "Country:", ctyOutDegree[items[-5][0]], " ", items[-5][0], "\n"

    if(edgetype.lower()=='in'):
        print "\nTop five countries people have to say about:"
        print "Country:", ctyInDegree[items[-1][0]], " ", items[-1][0]
        print "Country:", ctyInDegree[items[-2][0]], " ", items[-2][0]
        print "Country:", ctyInDegree[items[-3][0]], " ", items[-3][0]
        print "Country:", ctyInDegree[items[-4][0]], " ", items[-4][0]
        print "Country:", ctyInDegree[items[-5][0]], " ", items[-5][0], "\n"

    if(edgetype.lower()=='undirected'):
        print "\nTop five interacted:"
        print "Country:", ctyInDegree[items[-1][0]], " ", items[-1][0]
        print "Country:", ctyInDegree[items[-2][0]], " ", items[-2][0]
        print "Country:", ctyInDegree[items[-3][0]], " ", items[-3][0]
        print "Country:", ctyInDegree[items[-4][0]], " ", items[-4][0]
        print "Country:", ctyInDegree[items[-5][0]], " ", items[-5][0], "\n"


    fig = plt.figure ()
    ax = fig.add_subplot (111)
    optimalbin=math.ceil(math.log(len(Nklist),2)+1)
    print "Optimal bin:", optimalbin
    # optimalbinwidth=   #width=samplesize/noofbins
    _n,_binsedges,_patches=ax.hist(Nklist, bins=binsize, normed=normed)
    # print "_n",_n, " size:", len(_n), " type:", type(_n)
    # print "_bins_edges",_binsedges, " size:", len(_binsedges)

    if overlayfit==True:
        #Cleansing zero valued results
        i=0
        droplist=[]
        for n in _n:
            if n==0.:
                droplist.append(i)
            i=i+1

        _finaln = np.delete(_n, droplist)
        _bins=_binsedges[1:]
        _finalbins=np.delete(_bins,droplist)


        print "_finaln", _finaln, " size:", len(_finaln)
        print "_finalbins", _finalbins, " size:", len(_finalbins)

        print "_patches",_patches

        print "log_bins",np.log(_finalbins)
        print "log__n",np.log(_finaln)
        # slope, intercept = np.polyfit(np.log(_finalbins),np.log(_finaln),1)
        # slope, intercept = np.polyfit((np.log(_finalbins)), np.log(_finaln), 1)
        slope, intercept = np.polyfit(((_finalbins)), (_finaln), 1)
        print "Slope:", slope
        plt.plot(_finalbins,slope*_finalbins+intercept,'r',linewidth=3)

        plresults=powerlaw.Fit(Nklist)
        # print ("ntail:",plresults.n_tail())
        # plresults.plot_pdf(color='r')
        plresults.power_law.plot_pdf(color='g', linestyle='--')
        # plresults2 = powerlaw.Fit(Nklist,xmin=1)
        # plresults2.plot_pdf(color='b', linestyle='--')
        ax.annotate(r'$\alpha$='+"{0:.2f}".format(plresults.alpha), xy=(10**2.5, 10**-2), xycoords='data',
                    xytext=(0.8, 0.95), textcoords='axes fraction',
                    horizontalalignment='right', verticalalignment='top',
                    )
        # plresults.plot_ccdf(color='g')
        print("Powerlaw alpha:",plresults.alpha)
        print("Powerlaw xmin:", plresults.xmin)

    title=edgetype.upper()+"-Degree Distribution"


    if(loglog==True):
        ax.set_xscale('log')
        ax.set_yscale('log')
        title=edgetype.upper()+"-Degree Distribution (Log/Log)"

    plt.xlabel('Degree')
    plt.ylabel('P(k)') if(normed==True) else plt.ylabel('Frequency')

    plt.title(title)
    plt.autoscale()
    plt.show()


def plot_weighted_degree_distribution_bined (graph, loglog=False, edgetype='undirected', binsize=10, normed=False) :
    """
    Plot the degree distribution of graph.
    :param graph:
    :param loglog: log log scale. (Used to access if power law exists)
    :param edgetype:undirected, in links only, out links only.
    :param binsize:
    :param normed: Outcome in terms of probability instead of counts.
    :return:
    """
    print "loglog=",loglog, "edgetype=",edgetype,"binsize=",binsize,"normalised=",normed
    totalNodes=float(len(graph.nodes()))
    print "Total no of countries in graph: ", totalNodes
    Nklist=[]
    degs = {} #K, Nk
    ctyOutDegree=defaultdict(str)
    ctyInDegree=defaultdict(str)
    ctyDegree = defaultdict(str)
    for n in graph.nodes () :
        deg=0
        if(edgetype=='undirected'):
            deg = graph.degree (n, 'weight')
            Nklist.append(graph.degree (nbunch=n,weight='weight'))
            ctyDegree[deg] = str(ctyDegree[deg]) + "," + str(n)
        elif(edgetype=='in'):
            deg = graph.in_degree(n, 'weight')
            Nklist.append(deg)
            # print "Country:", n, "  In-Degree:", deg
            ctyInDegree[deg]=str(ctyInDegree[deg])+","+str(n)
        elif(edgetype=='out'):
            deg = graph.out_degree(n, 'weight')
            Nklist.append(graph.out_degree(n, 'weight'))
            # print "Country:", n, "  Out-Degree:", deg
            ctyOutDegree[deg] = str(ctyOutDegree[deg]) + "," + str(n)
        if deg not in degs:
            degs[deg] = 0
        degs[deg] += 1



    items = sorted(degs.items())

    if(edgetype.lower()=='out'):
        print "\nTop five busybodies:"
        print "Country:", ctyOutDegree[items[-1][0]], " ", items[-1][0]
        print "Country:", ctyOutDegree[items[-2][0]], " ", items[-2][0]
        print "Country:", ctyOutDegree[items[-3][0]], " ", items[-3][0]
        print "Country:", ctyOutDegree[items[-4][0]], " ", items[-4][0]
        print "Country:", ctyOutDegree[items[-5][0]], " ", items[-5][0], "\n"

    if(edgetype.lower()=='in'):
        print "\nTop five countries people have to say about:"
        print "Country:", ctyInDegree[items[-1][0]], " ", items[-1][0]
        print "Country:", ctyInDegree[items[-2][0]], " ", items[-2][0]
        print "Country:", ctyInDegree[items[-3][0]], " ", items[-3][0]
        print "Country:", ctyInDegree[items[-4][0]], " ", items[-4][0]
        print "Country:", ctyInDegree[items[-5][0]], " ", items[-5][0], "\n"

    if(edgetype.lower()=='undirected'):
        print "\nTop five interacted:"
        print "Country:", ctyInDegree[items[-1][0]], " ", items[-1][0]
        print "Country:", ctyInDegree[items[-2][0]], " ", items[-2][0]
        print "Country:", ctyInDegree[items[-3][0]], " ", items[-3][0]
        print "Country:", ctyInDegree[items[-4][0]], " ", items[-4][0]
        print "Country:", ctyInDegree[items[-5][0]], " ", items[-5][0], "\n"


    fig = plt.figure ()
    ax = fig.add_subplot (111)
    ax.hist(Nklist, bins=binsize, normed=normed)
    title=edgetype.upper()+"-Degree Distribution"


    if(loglog==True):
        ax.set_xscale('log')
        ax.set_yscale('log')
        title=edgetype.upper()+"-Degree Distribution (Log/Log)"

    plt.xlabel('Degree')
    plt.ylabel('P(k)') if(normed==True) else plt.ylabel('Frequency')

    plt.title(title)
    plt.show()

def loadedgelistaslist(csvedgefile):
    edgelist=[]
    with open(csvedgefile, 'r') as csvfile:
        edgereader = csv.reader(csvfile, delimiter=',')
        for edge in edgereader:
            src=edge[0]
            tgt=edge[1]
            edgelist.append([src,tgt])
    return edgelist



def smallworldstats(graph):
    avgclustering=nx.average_clustering(graph)
    avgpathlength=nx.average_shortest_path_length(graph)
    return avgclustering, avgpathlength


def prunegraph(graph,no_prunelist=[]):
    """
    To prune the graph down to selected countries. Affect edges will be removed too
    :param graph:
    :param prunelist:
    :return:
    """
    count=0
    totalNodes=len(graph.nodes())
    for cty in graph.nodes():
        if cty not in no_prunelist:
            graph.remove_node(cty)
            count=count+1
    print "Total nodes: ", totalNodes, " nodes removed: ", count, " nodes left: ", len(graph.nodes())
    return graph


def convertweightedtosignedgraph(weightdigraph):
    """
    Converted directed weighted graph into signed undirected graph
    For the 2 edges (if any) between 2 nodes, add up to form the sign.
    Supply warning if edges has opposing signs.
    :param weightdigraph:
    :return:
    """
    undirectedG=nx.Graph()
    edgearray=defaultdict(int)
    for node in weightdigraph.nodes():  #For each node
        for neighbournode in weightdigraph.neighbors(node):  #Get its neighbours
            if edgearray[node+neighbournode]==0 and edgearray[neighbournode+node]==0: #Ensure not processed before
                if weightdigraph.has_edge(node,neighbournode):
                    weightSrcToTgt=weightdigraph[node][neighbournode]['weight']
                else:
                    weightSrcToTgt=0
                if weightdigraph.has_edge(neighbournode,node):
                    weightTgtToSrc = weightdigraph[neighbournode][node]['weight']
                else:
                    weightTgtToSrc=0
                finalweight=0
                if (weightSrcToTgt<0 and weightTgtToSrc>0) or (weightSrcToTgt>0 and weightTgtToSrc<0):
                    print "Different sign!!  Node:",node, " Node:",neighbournode
                    finalweight=np.sign(weightSrcToTgt+weightTgtToSrc)
                else:
                    finalweight = np.sign(weightSrcToTgt)

                edgearray[node+neighbournode]=finalweight
                edgearray[neighbournode+node] = finalweight
                undirectedG.add_node(node)
                undirectedG.add_node(neighbournode)
                undirectedG.add_edge(node,neighbournode,weight=finalweight)

    return undirectedG







def balancecheckoneyear(counter=[],focus=[],nameofpruned='',prefixoriginal=''):
    """
    Structural balance conversion to simple signed graph
    :param counter:
    :param focus:
    :param nameofpruned:
    :param prefixoriginal:
    :return:
    """
    ctyarray=loadUNCountryList('UNCountryCode.txt')

    for count in counter:
        focus = focus
        weightcsv_original='2016' + str(count)+prefixoriginal+'.txt'
        cleansedweightedcsv = '2016' + str(count) + '_WeightedCleansed.txt'
        prunedweightcsv='2016' + str(count) + '_WeightedCleansedPruned_'+nameofpruned+'.txt'
        prunedweightsignedcsv = '2016' + str(count) + '_WeightedCleansedPruned_' + nameofpruned + '_signed.txt'

        cleanseWeightedEdgeData(csvfile=weightcsv_original,cleansedcsvfile=cleansedweightedcsv, ctyarray=ctyarray)
        dG = nx.read_weighted_edgelist(cleansedweightedcsv, delimiter=',', create_using=nx.DiGraph())
        prunedgraph = prunegraph(dG, no_prunelist=focus)
        saveWeightedGraphAsCSV(graph=prunedgraph, tocsvfile=prunedweightcsv)
        # saveGraphAsCSV(graph=sea_graph, tocsvfile=seacsv)

        signedgraph = convertweightedtosignedgraph(prunedgraph)
        saveWeightedGraphAsCSV(graph=signedgraph, tocsvfile=prunedweightsignedcsv)



def checkSuccessorCorrelation(digraph):
    """
    Check if successor node is actually predecessor node
    :param digraph:
    :return:
    """
    succesorlist=defaultdict(int)
    predecessorList=defaultdict(int)
    for node in digraph.nodes():  # For each node
        for successor in digraph.successors(node):
            succesorlist[node+"_"+successor]=1
        for predecessor in digraph.predecessors(node):
            predecessorList[node+"_"+predecessor]=1

    hitratethreshold=0.8
    successorThreshold=30
    totalNodesWithMoreThanThresholdSuccesor=0
    totalNodesProcessed=0
    higherthanthreshold=0
    for node in digraph.nodes():  # For each node
        totalSucessors=len(digraph.successors(node))
        totalPredecessors=len(digraph.predecessors(node))
        if (totalSucessors>successorThreshold):
            totalNodesWithMoreThanThresholdSuccesor+=1
            hitCount=0.0
            for successor in digraph.successors(node):
                if predecessorList[node+"_"+successor]==1:
                    hitCount+=1
            if (totalSucessors>0):
                totalNodesProcessed+=1
                print ("Node: ", node, " Successors:Predecessor"+str(totalSucessors)+":"+str(totalPredecessors)+ "  Sucessor is Predecessor count=", hitCount )
                if((hitCount/totalSucessors)>hitratethreshold):
                    higherthanthreshold+=1
    print (str(higherthanthreshold) + " out of " +  str(totalNodesWithMoreThanThresholdSuccesor) + " nodes, or " + str(higherthanthreshold/totalNodesWithMoreThanThresholdSuccesor) +" has higher then " + str(hitratethreshold))


def checkPredecessorCorrelation(digraph):
    """
    Check if successor node is actually predecessor node
    :param digraph:
    :return:
    """
    succesorlist=defaultdict(int)
    predecessorList=defaultdict(int)
    for node in digraph.nodes():  # For each node
        for successor in digraph.successors(node):
            succesorlist[node+"_"+successor]=1
        for predecessor in digraph.predecessors(node):
            predecessorList[node+"_"+predecessor]=1

    hitratethreshold=0.8
    predecessorThreshold=30
    totalNodesWithMoreThanThresholdPredecessors=0
    totalNodesProcessed=0
    higherthanthreshold=0
    for node in digraph.nodes():  # For each node
        totalSucessors=len(digraph.successors(node))
        totalPredecessors=len(digraph.predecessors(node))
        if (totalPredecessors>predecessorThreshold):
            totalNodesWithMoreThanThresholdPredecessors+=1

            hitCount=0.0
            for predecessor in digraph.predecessors(node):
                if succesorlist[node+"_"+predecessor]==1:
                    hitCount+=1
            if (totalPredecessors>0):
                totalNodesProcessed+=1
                print ("Node: ", node, " Predecessor:Successors"+str(totalPredecessors)+":"+str(totalSucessors)+ "   Predecessor is Sucessor count=", hitCount )
                if((hitCount/totalPredecessors)>hitratethreshold):
                    higherthanthreshold+=1
    print (str(higherthanthreshold) + " out of " +  str(totalNodesWithMoreThanThresholdPredecessors) + " nodes, or " + str(higherthanthreshold/totalNodesWithMoreThanThresholdPredecessors) +" has higher then " + str(hitratethreshold))

def create_randomgraph(_n=0,_m=0, _directed=False):
    """
    Generate ER random graph
    :param _n:
    :param _m:
    :param _directed:
    :return:
    """
    n = _n
    m = _m
    directed=_directed

    G = nx.gnm_random_graph(n, m)
    print ("NoOfNodes:",nx.number_of_nodes(G))
    print ("NoOfEdges:", nx.number_of_edges(G))
    print ("Avg shortest path:",nx.average_shortest_path_length(G))
    # print ("Avg clustering coefficient:", nx.average_clustering(G))
    return G



csvfile = '201608stripped.txt'
weightedcsvfile='201608WeightedStripped.txt'
cleansedcsvfile = 'Final201608edge.txt'
weightedcleansedcsvfile='201608weightedcleansedstripped.txt'
ctyarray=loadUNCountryList('UNCountryCode.txt')

##One time operation, once file written, no need to worry about this anymore.
# writeUNCountryList(ctyarray)
# cleanseEdgeData(csvfile=csvfile,cleansedcsvfile=cleansedcsvfile, ctyarray=ctyarray)



# # 11 months signed graph for TOP 4 in world
# counter=['01','02','03','04','05','06','07','08','09','10','11','11']
# focus=['USA','GBR','CHN','RUS']
# balancecheckoneyear(counter=counter,focus=focus,nameofpruned='WORLD4',prefixoriginal='_GOV_stripped')



# #11 months signed graph for SEA top 3
# counter=['01','02','03','04','05','06','07','08','09','10','11']
# focus=['IDN','SGP','MYS']
# balancecheckoneyear(counter=counter,focus=focus,nameofpruned='SEA4_GOV',prefixoriginal='_GOV_stripped')



# ### CHECK IF IN NODE = OUT NODES
# dG = nx.read_weighted_edgelist('201608_WeightedCleansed.txt', delimiter=',', create_using=nx.DiGraph())
# checkSuccessorCorrelation(dG)
# checkPredecessorCorrelation(dG)


#Print basic info
# G=nx.read_edgelist(cleansedcsvfile,delimiter=',',create_using=nx.DiGraph())
# print "Nodes:", nx.number_of_nodes(G)
# print "Edges:", nx.number_of_edges(G)
# print ("Avg shortest path:", nx.average_shortest_path_length(G))
# print ("Avg clustering coefficient:", nx.average_clustering(G))


# G=nx.read_edgelist(cleansedcsvfile,delimiter=',',create_using=nx.DiGraph())
# print "Nodes:", nx.number_of_nodes(G)
# print "Edges:", nx.number_of_edges(G)
# print ("Avg shortest path:", nx.average_shortest_path_length(G))
# print ("Avg clustering coefficient:", nx.average_clustering(G))


#Plot Degree Distribution
G=nx.read_edgelist(cleansedcsvfile,delimiter=',',create_using=nx.DiGraph())
plot_degree_distribution_bined(graph=G, loglog=False, edgetype='in', binsize=13, normed=True, overlayfit=False)
plot_degree_distribution_bined(graph=G, loglog=False, edgetype='out', binsize=13, normed=True, overlayfit=False)
plot_degree_distribution_bined(graph=G, loglog=True, edgetype='in', binsize=20, normed=True, overlayfit=True)
plot_degree_distribution_bined(graph=G, loglog=True, edgetype='out', binsize=20, normed=True, overlayfit=True)



#Random VS Opinion Network
# rDG=create_randomgraph(198,6989, True)
# saveGraphAsCSV(rDG, 'rDG.txt')