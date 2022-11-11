import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



report = "./report32.csv"


df = pd.read_csv(report)


plt.figure()
plt.plot(df["epoch"],df["train_ssim"], label="Train")
plt.plot(df["epoch"],df["test_ssim"], label="Test")
plt.legend()
plt.savefig(report+".png")
plt.close()