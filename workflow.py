from easydict import EasyDict
import tensorflow as tf

from model import Word2VecModel

class Parameter:
    epochs = 0
    batch_size = 0
    shuffle_buffer_size = 0

class Word2VecWorkflow:
    def __init__(self, model: Word2VecModel, param: dict) -> None:
        self.param = EasyDict(param)
        self.model = model
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(0.01, 10, 0.99)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr_scheduler)
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")
    
    @tf.function
    def train_step(self, words_in, words_out):
        with tf.GradientTape() as tape:
            pred = self.model(words_in)
            loss = self.loss_object(words_out, pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(words_out, pred)

    @tf.function
    def test_step(self, words_in, words_out):
        pred = self.model(words_in)
        loss = self.loss_object(words_out, pred)
        self.test_loss(loss)
        self.test_accuracy(words_out, pred)

    def train(self, x_train, y_train, x_test, y_test):
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(self.param.shuffle_buffer_size).batch(self.param.batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.param.batch_size)
        for epoch in range(self.param.epochs):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            for words_in, words_out in train_ds:
                self.train_step(words_in, words_out)
            
            for words_in, words_out in test_ds:
                self.test_step(words_in, words_out)
            
            self.summarize_train(epoch)
            self.model.save_weights("kite_word2vec.h5")
    
    def test(self, x_test, y_test):
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.param.batch_size)
        for words_in, words_out in test_ds:
            self.test_step(words_in, words_out)
        self.summarize_test()
    
    def summarize_train(self, epoch):
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Learning Rate: {}'
        print(template.format(
            epoch + 1,
            self.train_loss.result(),
            self.train_accuracy.result() * 100,
            self.train_loss.result(),
            self.test_accuracy.result() * 100,
            self.optimizer._decayed_lr(tf.float32)))

    def summarize_test(self):
        template = 'Test Loss: {}, Test Accuracy: {}'
        print(template.format(
            self.test_loss.result(),
            self.test_accuracy.result() * 100))