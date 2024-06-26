import sys
import json
import matplotlib.pyplot as plt

historyDictFilePath = str(sys.argv[1])
with open(historyDictFilePath, "r") as historyDictFile:
  historyDict = json.load(historyDictFile)

epochs = int(sys.argv[2])
epochRange = range(1,epochs+1)

historyLoss = historyDict["loss"]
historyRecall = historyDict["recall"]
historyValLoss = historyDict["val_loss"]
historyValRecall = historyDict["val_recall"]

plt.plot(epochRange, historyLoss, color="r", label="loss")
plt.plot(epochRange, historyRecall, color="y", label="recall")
plt.plot(epochRange, historyValLoss, color="g", label="val_loss")
plt.plot(epochRange, historyValRecall, color="b", label="val_recall")

plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Performance")
plt.legend()

plt.savefig("historyDict.pdf")

