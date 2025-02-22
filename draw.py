from keras.src.saving import load_model
def loadmodel():
    model = load_model('efficientnetb0.h5')
    print(model.summary())


if __name__ == '__main__':
    loadmodel()