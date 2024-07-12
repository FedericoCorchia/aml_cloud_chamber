import sys
import json
import matplotlib.pyplot as plt

historyDictFilePath = str(sys.argv[1])
with open(historyDictFilePath, "r") as historyDictFile:
  historyDict = json.load(historyDictFile)
plotOutputFilePath = str(sys.argv[2])

epochs = int(sys.argv[3])
epochRange = range(1,epochs+1)

historyLoss = historyDict["loss"]
historyPrecision = historyDict["precision"]
historyRecall = historyDict["recall"]
historyValLoss = historyDict["val_loss"]
historyValPrecision = historyDict["val_precision"]
historyValRecall = historyDict["val_recall"]

plt.plot(epochRange, historyLoss, color="r", label="loss", linestyle="--")
plt.plot(epochRange, historyPrecision, color="g", label="precision", linestyle="--")
plt.plot(epochRange, historyRecall, color="b", label="recall", linestyle="--")
plt.plot(epochRange, historyValLoss, color="r", label="val_loss")
plt.plot(epochRange, historyValPrecision, color="g", label="val_precision")
plt.plot(epochRange, historyValRecall, color="b", label="val_recall")

plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Performance")
plt.legend()

plt.savefig(plotOutputFilePath)

