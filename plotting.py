import sys
import json
import matplotlib.pyplot as plt

#load training performance metrics from json file
historyDictFilePath = str(sys.argv[1])
with open(historyDictFilePath, "r") as historyDictFile:
  historyDict = json.load(historyDictFile)
plotOutputFilePath = str(sys.argv[2])

#plotting
epochs = 20
epochRange = range(1,epochs+1)

historyTrainLoss = historyDict["loss"]
historyTrainPrecision = historyDict["precision"]
historyTrainRecall = historyDict["recall"]
historyValidLoss = historyDict["val_loss"]
historyValidPrecision = historyDict["val_precision"]
historyValidRecall = historyDict["val_recall"]

plt.plot(epochRange, historyTrainLoss, color="r", label="Train Loss", linestyle="--")
plt.plot(epochRange, historyTrainPrecision, color="g", label="Train Precision", linestyle="--")
plt.plot(epochRange, historyTrainRecall, color="b", label="Train Recall", linestyle="--")
plt.plot(epochRange, historyValidLoss, color="r", label="Valid Loss")
plt.plot(epochRange, historyValidPrecision, color="g", label="Valid Precision")
plt.plot(epochRange, historyValidRecall, color="b", label="Valid Recall")

plt.xlabel("Epoch")
plt.xticks(epochRange)
plt.ylabel("Value")
plt.title("Training Performance")
plt.legend()

#saving plot to file
plt.savefig(plotOutputFilePath)

