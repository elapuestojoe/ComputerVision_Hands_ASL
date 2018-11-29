#Plot values of Xception model
import matplotlib.pyplot as plt 

x = [1, 2, 3]
categorical_accuracy = [0.4889, 0.9385, 0.9655]
test_accuracy = [0.8464, 0.8577, 0.9176]

plt.plot(x, categorical_accuracy, label="Training accuracy")
plt.plot(x, test_accuracy, label="Testing accuracy")
plt.title("Model performance")
plt.ylabel("Accuracy")
plt.xlabel("epoch")
plt.legend()
plt.show()