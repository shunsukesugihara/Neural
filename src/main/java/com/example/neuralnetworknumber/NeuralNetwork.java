package com.example.neuralnetworknumber;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A simple feedforward neural network
 * It includes one hidden layer and uses backpropagation to adjust its weights.
 * @author Shunsuke Sugihara
 * @since 5/10/2025
 */



public class NeuralNetwork {
    private final List<List<Neuron>> layers = new ArrayList<>();
    private static final double ETA = 0.15;


    private static final double ALPHA = 0.5;

    /**
     * Creates the neural network with the 3 layers
     * 784 input neurons, 64 hidden neurons, 10 output neurons
     */

    public NeuralNetwork() {
        int[] topology = {784, 64, 10};
        for (int layer = 0; layer < topology.length; layer++) {
            List<Neuron> layerNeurons = new ArrayList<>();
            int numOutputs = (layer == topology.length - 1) ? 0 : topology[layer + 1];
            for (int i = 0; i <= topology[layer]; i++) {
                layerNeurons.add(new Neuron(numOutputs, i));
            }


            layerNeurons.get(layerNeurons.size() - 1).setOutputVal(1.0);
            layers.add(layerNeurons);
        }
    }


    /**
     * Passes the input values through the network to calculate the output.
     *
     * @param inputVals A list of pixel values from the input image.
     */

    public void feedForward(List<Double> inputVals) {
        for (int i = 0; i < inputVals.size(); i++) {
            layers.get(0).get(i).setOutputVal(inputVals.get(i));
        }

        for (int layerNum = 1; layerNum < layers.size(); layerNum++) {
            List<Neuron> prevLayer = layers.get(layerNum - 1);
            for (int n = 0; n < layers.get(layerNum).size() - 1; n++) {
                layers.get(layerNum).get(n).feedForward(prevLayer);
            }
        }
    }

    /**
     * Updates the weights of the network based on the difference between
     * the prediction and the correct answer using backpropagation.
     *
     * @param targetVals The correct value typed in
     */

    public void backProp(List<Double> targetVals) {
        List<Neuron> outputLayer = layers.get(layers.size() - 1);


        for (int n = 0; n < outputLayer.size() - 1; n++) {
            outputLayer.get(n).calcOutputGradients(targetVals.get(n));
        }


        for (int layerNum = layers.size() - 2; layerNum > 0; layerNum--) {
            List<Neuron> hiddenLayer = layers.get(layerNum);
            List<Neuron> nextLayer = layers.get(layerNum + 1);

            for (Neuron neuron : hiddenLayer) {
                neuron.calcHiddenGradients(nextLayer);
            }
        }


        for (int layerNum = layers.size() - 1; layerNum > 0; layerNum--) {
            List<Neuron> layer = layers.get(layerNum);
            List<Neuron> prevLayer = layers.get(layerNum - 1);

            for (int n = 0; n < layer.size() - 1; n++) {
                layer.get(n).updateInputWeights(prevLayer);
            }
        }
    }

    /**
     * Returns the output values of the last layer of the network.
     *
     * @return A list of output neuron activation from 0 to 1.
     */

    public List<Double> getResults() {
        List<Double> results = new ArrayList<>();
        for (int n = 0; n < layers.get(layers.size() - 1).size() - 1; n++) {
            results.add(layers.get(layers.size() - 1).get(n).getOutputVal());
        }
        return results;
    }

    /**
     * Takes the raw image input and processes it through the network,
     * and returns the prediction
     *
     * @param input The 784 grayscale valeus
     * @return The neuron with the highest activation.
     */

    public int predict(double[] input) {
        List<Double> inputVals = new ArrayList<>();
        for (double val : input) inputVals.add(val / 255.0);
        feedForward(inputVals);
        List<Double> outputs = getResults();
        int maxIndex = 0;
        for (int i = 1; i < outputs.size(); i++) {
            if (outputs.get(i) > outputs.get(maxIndex)) maxIndex = i;
        }
        return maxIndex;
    }

    /**
     * A single neuron in the network.
     * Stores its activation value, gradient, and connections
     */

    static class Neuron {
        private static final Random rand = new Random();
        private final List<Connection> outputWeights = new ArrayList<>();
        private final int index;
        private double outputVal;
        private double gradient;

        /**
         * Initializes a neuron with a set number wuth connections.
         * Each connection has a random weigh that will be updated as the training progresses
         *
         * @param numOutputs Number of neurons in the next layer.
         * @param index Position of the neuron in its layer.
         */

        public Neuron(int numOutputs, int index) {
            this.index = index;
            for (int i = 0; i < numOutputs; i++) {
                outputWeights.add(new Connection(rand.nextDouble(), 0.0));
            }
        }

        /**
         * Seta the activation value of the neuron.
         *
         * @param val Activation value between 0 to 1
         */

        public void setOutputVal(double val) {
            outputVal = val;
        }

        /**
         * Returns the activation value of the neuron.
         *
         * @return The neuron's output value.
         */

        public double getOutputVal() {
            return outputVal;
        }

        /**
         * Calculates the neuron's output from the weighted sum of previous layer,
         * and pass it through an activation function.
         *
         * @param prevLayer The neurons from the previous layer.
         */

        public void feedForward(List<Neuron> prevLayer) {
            double sum = 0.0;
            for (Neuron neuron : prevLayer) {
                sum += neuron.getOutputVal() * neuron.outputWeights.get(index).weight;
            }
            outputVal = activationFunction(sum);
        }

        /**
         * Calculates the gradient for an output neuron.
         *
         * @param targetVal The correct value from the training data
         */

        public void calcOutputGradients(double targetVal) {
            double delta = targetVal - outputVal;
            gradient = delta * activationFunctionDerivative(outputVal);
        }

        /**
         * Calculates the gradient for the hidden neuron based on the next layers gradients
         *
         * @param nextLayer The neurons in the next laye
         */

        public void calcHiddenGradients(List<Neuron> nextLayer) {
            double dow = 0.0;
            for (int n = 0; n < nextLayer.size() - 1; n++) {
                dow += outputWeights.get(n).weight * nextLayer.get(n).gradient;
            }
            gradient = dow * activationFunctionDerivative(outputVal);
        }

        /**
         * Updates the input weights from the previous layer using the calculated gradient.
         * Includes learning rate and momentum.
         *
         * @param prevLayer The neurons of the previous layer
         */

        public void updateInputWeights(List<Neuron> prevLayer) {
            for (Neuron neuron : prevLayer) {
                Connection conn = neuron.outputWeights.get(index);
                double oldDeltaWeight = conn.deltaWeight;
                double newDeltaWeight = ETA * neuron.getOutputVal() * gradient + ALPHA * oldDeltaWeight;
                conn.deltaWeight = newDeltaWeight;
                conn.weight += newDeltaWeight;
            }
        }

        private double activationFunction(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }

        private double activationFunctionDerivative(double x) {
            return x * (1.0 - x);
        }
    }

    /**
     * The connection between　the  two neurons,
     * including the weight and the most recent change in weight.
     */

    static class Connection {
        double weight;
        double deltaWeight;

        /**
         * Creates a connection with a the given weight and previous delta value.
         *
         * @param weight The initial connection weight.
         * @param deltaWeight The previous change in weight momentum
         */

        public Connection(double weight, double deltaWeight) {
            this.weight = weight;
            this.deltaWeight = deltaWeight;
        }
    }
}