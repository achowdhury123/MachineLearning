# The following code first initializes a neural network that was trained on a dataset
# of right and left directional arrows to classify between these two images. The neural
# network is then used to control a robotic car, with the help of a Raspberry Pi Camera
# and an ultrasonic motion sensor. When the ultrasonic motion sensor detects an object
# nearby, the car stops. The Raspberry Pi Camera then takes a picture, and the neural
# network determinse whether the picture is a right or left direcitonal arrow. The car
# then adjusts its course based on whether the image is classified as right or left.

import numpy
import scipy.special
import scipy.misc
import scipy.ndimage
import pickle
from picamera import PiCamera
from time import sleep
from PIL import Image
import RPi.GPIO as GPIO
import time

class neural_network:

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.input_hidden_weights = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.hidden_output_weights = numpy.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

    def sigmoid_function(self, inputs):
        outputs = 1/(1 + numpy.exp(-inputs))
        return outputs

    def forward_propogation(self, inputs):
        hidden_node_inputs = numpy.dot(self.input_hidden_weights, inputs)
        hidden_node_outputs = self.sigmoid_function(hidden_node_inputs)
        output_node_inputs = numpy.dot(self.hidden_output_weights, hidden_node_outputs)
        outputs = self.sigmoid_function(output_node_inputs)
        return outputs, hidden_node_outputs

    def train(self, learning_rate, training_inputs, expected_outputs, iterations, batch_size):
        training_inputs = numpy.array(training_inputs, ndmin = 2)
        expected_outputs = numpy.array(expected_outputs, ndmin = 2)
        for iterations in range(iterations):
            for i in range(0, training_inputs.shape[0], batch_size):
                input_training_batch = training_inputs[i:i + batch_size]
                expected_outputs_batch = expected_outputs[i:i + batch_size]
                hidden_output_weight_gradient = numpy.zeros((self.output_nodes, self.hidden_nodes))
                input_hidden_weight_gradient = numpy.zeros((self.hidden_nodes, self.input_nodes))
                for rows in range(input_training_batch.shape[0]):
                    input_row = numpy.array(input_training_batch[rows], ndmin = 2).T
                    expected_output_row = numpy.array(expected_outputs_batch[rows], ndmin = 2).T
                    output_row, hidden_layer_row = self.forward_propogation(input_row)
                    print(expected_output_row, output_row)
                    output_error = expected_output_row - output_row
                    hidden_layer_error = numpy.dot(self.hidden_output_weights.T, output_error)
                    hidden_output_weight_gradient+=learning_rate*numpy.dot(output_error*output_row*(1-output_row), numpy.transpose(hidden_layer_row))
                    input_hidden_weight_gradient+=learning_rate*numpy.dot(hidden_layer_error*hidden_layer_row*(1-hidden_layer_row), numpy.transpose(input_row))
                self.hidden_output_weights+=hidden_output_weight_gradient
                self.input_hidden_weights+=input_hidden_weight_gradient
        pass

    def test(self, inputs):
        inputs = numpy.array(inputs, ndmin = 2).T
        outputs, hidden_layer_row = self.forward_propogation(inputs)
        return outputs

pickle_in = open('direction_classifier.pickle', 'rb')
direction_classifier = pickle.load(pickle_in)

camera = PiCamera()

camera.rotation = 180

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(True)

Left_Motor_Forward = 16
Left_Motor_Backward = 18

Right_Motor_Forward = 11
Right_Motor_Backward = 13

ECHO = 38
TRIG = 40

GPIO.setup(Left_Motor_Forward,GPIO.OUT)
GPIO.setup(Left_Motor_Backward,GPIO.OUT)

GPIO.setup(Right_Motor_Forward,GPIO.OUT)
GPIO.setup(Right_Motor_Backward,GPIO.OUT)

GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO,GPIO.IN)

def forward():
    GPIO.output(Left_Motor_Forward,GPIO.HIGH)
    GPIO.output(Left_Motor_Backward,GPIO.LOW)

    GPIO.output(Right_Motor_Forward,GPIO.HIGH)
    GPIO.output(Right_Motor_Backward,GPIO.LOW)

def backward():
    GPIO.output(Left_Motor_Forward,GPIO.LOW)
    GPIO.output(Left_Motor_Backward,GPIO.HIGH)

    GPIO.output(Right_Motor_Forward,GPIO.LOW)
    GPIO.output(Right_Motor_Backward,GPIO.HIGH)

def right():
    GPIO.output(Left_Motor_Forward,GPIO.HIGH)
    GPIO.output(Left_Motor_Backward,GPIO.LOW)

    GPIO.output(Right_Motor_Forward,GPIO.LOW)
    GPIO.output(Right_Motor_Backward,GPIO.LOW)

def left():
    GPIO.output(Left_Motor_Forward,GPIO.LOW)
    GPIO.output(Left_Motor_Backward,GPIO.HIGH)

    GPIO.output(Right_Motor_Forward,GPIO.LOW)
    GPIO.output(Right_Motor_Backward,GPIO.LOW)

def stop():
    GPIO.output(Left_Motor_Forward,GPIO.LOW)
    GPIO.output(Left_Motor_Backward,GPIO.LOW)

    GPIO.output(Right_Motor_Forward,GPIO.LOW)
    GPIO.output(Right_Motor_Backward,GPIO.LOW)

while True:
    GPIO.output(TRIG,True)
    time.sleep(0.00001)
    GPIO.output(TRIG,False)
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
    pulse_duration = pulse_end - pulse_start
    print(pulse_duration)
    distance = pulse_duration * 17150
    distance = round(distance, 2)
##    print(distance)
    if distance < 17:
        stop()
        time.sleep(2)
        camera.capture('/home/pi/image.png')
        direction_arrow = scipy.misc.imread('image.png', flatten = True)
        direction_arrow = scipy.misc.imresize(direction_arrow, (28, 28))
        direction_arrow = direction_arrow.reshape(784)
        direction_arrow = direction_arrow/255*0.99+0.01
        direction = numpy.array(direction_arrow, ndmin = 2)
        x = direction_classifier.test(direction_arrow)
        if x[0] < x[1]:
            right()
            time.sleep(0.75)
            print('right')
            stop()
            time.sleep(0.5)
        else:
            left()
            time.sleep(0.75)
            print('left')
            stop()
            time.sleep(0.5)
    else:
        forward()

GPIO.cleanup()
