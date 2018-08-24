import math
import numpy as np
import time
import mkl
import ctypes
import random
import threading

class myThread (threading.Thread):

   def __init__(self, threadID, name, posExample, negExample, step):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.posExample = posExample
      self.negExample = negExample
      self.step = step

   def run(self):
      #print("Starting {0}".format(self.name))
      step_gradient(self.posExample, self.negExample, self.step)
      #print("Exiting {0}".format(self.name))

# Global Model
current_w = math.sqrt(1/150)*np.random.random_sample(300)

# Helper function
def dotProduct(a, b):
    total = 0
    for i in range(0, len(a)):
        total += (a[i] + b[i])
    return total

# Helper function
def subtractLists(a, b):
    result = [0 for x in range(300)]
    for i in range(0, len(a)):
        result[i] = a[i] - b[i]
    return result

# Computes the error according to a given w
def compute_error(positives, negatives):
    totalError = 0.0

    # Positives
    posLinAlg = positives.dot(current_w)
    classValue = 1

    posLinAlg = posLinAlg*(-classValue)
    posLinAlg = np.exp(posLinAlg)
    posLinAlg = np.log(1 + posLinAlg)
    totalError += np.sum(posLinAlg)
    

    negLinAlg = negatives.dot(current_w)
    classValue = -1

    negLinAlg = negLinAlg*(-classValue)
    negLinAlg = np.exp(negLinAlg)
    negLinAlg = np.log(1 + negLinAlg)
    totalError += np.sum(negLinAlg)

    #Positives
    #posDots = positives.dot(w)
    #classValue = 1
    #for i in range(0, len(positives)):
    #    ePart = math.exp(-classValue * posDots[i])
    #    totalError += ( math.log( 1 +  ePart ) )

    # Negatives
    #classValue = -1
    ##negDots = negatives.dot(w)
    #for i in range(0, len(negatives)):
    #    ePart = math.exp(-classValue * negDots[i])
    #    totalError += ( math.log( 1 +  ePart ) )

    return totalError

def step_gradient(pos, neg, step):

    # Refer to global variable, current_w
    global current_w

    classValue = 1
    posLinAlg = pos.dot(current_w)
    posLinAlg = -classValue * posLinAlg
    posLinAlg = np.exp(posLinAlg)
    posLinAlg = (posLinAlg)/(1 + posLinAlg)
    posLinAlg = -classValue * posLinAlg

    w_pos_gradient = posLinAlg.dot(pos)


    classValue = -1
    negLinAlg = neg.dot(current_w)
    negLinAlg = -classValue * negLinAlg
    negLinAlg = np.exp(negLinAlg)
    negLinAlg = (negLinAlg)/(1 + negLinAlg)
    negLinAlg = -classValue * negLinAlg

    w_neg_gradient = negLinAlg.dot(neg)

    current_w = current_w - (step * (w_pos_gradient + w_neg_gradient))

def gradient_descent(positives, negatives, step, decay, iterations, batch, numThreads):

    # Thread List
    threads = np.empty(numThreads, myThread)

    # Step along the gradient 'iterations' amount of times

    start_time = time.time()

    for i in range(iterations):

        # Assign jobs to all threads
        for j in range(numThreads):
            posIndex = random.randint(0,1731-int(batch/2))
            negIndex = random.randint(0,57514-int(batch/2))
            pos = positives[posIndex:posIndex+int(batch/2)]
            neg = negatives[negIndex:negIndex+int(batch/2)]
            tempThread = myThread(j, "Thread-"+str(j), pos, neg, step)
            threads[j] = tempThread
            threads[j].start()
        
        # Wait for threads to finish
        for t in range(numThreads):
            threads[j].join()
        
        #print("Done threading")
        step -= decay

        print("Iteration number: {0}, Error: {1}, Elapsed time: {2}".format(i, format(compute_error(positives, negatives), '.2f'), format(time.time()-start_time, '.2f')))

def predictions(positives, negatives):

    posPredicts = positives.dot(current_w)
    negPredicts = negatives.dot(current_w)

    correctPos = np.sum(posPredicts > 0)
    correctNeg = np.sum(negPredicts < 0)

    print()

    print("Correct predicted {0} positives out of {1}!".format(correctPos, len(positives)))
    print("Success Rate: {0}".format(correctPos/len(positives)))

    print()

    print("Correct predicted {0} negatives out of {1}!".format(correctNeg, len(negatives)))
    print("Success Rate: {0}".format(correctNeg/len(negatives)))

    print()

def loadData():

    myFile = open('w8a.txt')

    # 57514 negatives, 1731 positives
    negatives = np.zeros((57514, 300), int)
    positives = np.zeros((1731, 300), int)

    negIndex = 0;
    posIndex = 0;

    for myLine in myFile:
        #print(lineNum)
        myPairs = myLine.split(" ")
        # Using pop() on first and last index
        # Since -1 is the first index and \n is the last index
        if(myPairs[0] == "-1"):
            myPairs.pop(0)
            myPairs.pop()
            for onePair in myPairs:
                splitPair = onePair.split(":");
                negatives[negIndex][int(splitPair[0])-1] = 1
            negIndex += 1

        elif(myPairs[0] == "+1"):
            myPairs.pop(0)
            myPairs.pop()
            tempRow = [0 for column in range(300)]
            for onePair in myPairs:
                splitPair = onePair.split(":");
                positives[posIndex][int(splitPair[0])-1] = 1
            posIndex += 1

    myFile.close()

    return [positives, negatives]

def main():

    # mkl_rt = ctypes.CDLL('mkl_rt')
    # print(np.show_config())
    # mkl.set_num_threads(2)
    # print(mkl.get_max_threads())

    # Set up data
    result = loadData()
    positives = result[0]
    negatives = result[1]

    # Set up parameters

    custom = input("Type in 'yes' to use custom settings, otherwise type anything else to use default settings\n")

    iterations = 100
    step = 0.005
    decay = (step/2)/iterations
    batch = 2
    numThreads = 10

    if(custom == 'yes'):
        iterations = int(input("Enter Iterations: "))
        print()
        step = float(input("Enter Step: "))
        print()
        decay = float(input("Enter Decay: "))
        print()
        batch = int(input("Enter Batch: "))
        print()
        numThreads = int(input("Enter Max Threads: "))
        print()
    
    print("Iterations: {0}\nStep: {1}\nDecay: {2}\nBatch: {3}\nMax Threads: {4}\n".format(iterations, step, decay, batch, numThreads))

    print("Starting Hogwild! GD")

    posTraining = positives[0:int(len(positives)*0.6)]
    negTraining = negatives[0:int(len(negatives)*0.6)]

    gradient_descent(posTraining, negTraining, step, decay, iterations, batch, numThreads)

    posPredictions = positives[int(len(positives)*0.6):len(positives)]
    negPredictions = negatives[int(len(negatives)*0.6):len(negatives)]

    predictions(posPredictions, negPredictions)

if  __name__ == '__main__':
    main()
