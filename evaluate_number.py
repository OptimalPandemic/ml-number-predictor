from keras.models import model_from_json
from keras.datasets import mnist
from matplotlib import pyplot as plt

# Load model
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
model = loaded_model
print("Loaded model from disk")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(10000, 28, 28, 1)

# Show image of each number and prediction
for i in range(10000):
    plt.imshow(x_test[i].reshape(28, 28), cmap='Greys')
    prediction = model.predict(x_test[i].reshape(1, 28, 28, 1))
    print("Your number is a: ", prediction.argmax())
    plt.show()
