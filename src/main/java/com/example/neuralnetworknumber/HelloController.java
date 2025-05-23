package com.example.neuralnetworknumber;

// HelloController.java

import javafx.fxml.FXML;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import java.util.Arrays;
import javafx.scene.control.Label;
import javafx.scene.input.MouseEvent;
import java.util.List;
import java.util.stream.Collectors;
import java.util.ArrayList;
import javafx.scene.control.ComboBox;
import javafx.scene.control.TextField;


/**
 * This class controls the interactions with the UI
 * Drawing on the canvas, training taking in the input.
 * @author Shunsuke Sugihara
 * @since 5/10/2025
 */

public class HelloController {
    @FXML
    private Canvas drawCanvas;
    @FXML
    private Button clearButton;
    @FXML
    private Button predictButton;
    @FXML
    private Label resultLabel;

    @FXML
    private ComboBox<Integer> labelSelector;
    @FXML
    private Button trainButton;

    @FXML
    private TextField labelInput;





    private GraphicsContext gc;

    private NeuralNetwork network = new NeuralNetwork();


    /**
     * initializes the canvas and buttons.
     * Calls automatically when the  UI is loaded.
     */
    @FXML
    public void initialize() {
        gc = drawCanvas.getGraphicsContext2D();
        clearCanvas();

        drawCanvas.setOnMouseDragged(this::handleDraw);

        clearButton.setOnAction(e -> clearCanvas());

        predictButton.setOnAction(e -> predictDigit());
        trainButton.setOnAction(e -> onTrainButtonClick());
    }

    /**
     * Called when the "Train" button is pressed.
     * The current drawing and the number typed by the user is sent
     * as a training example to improve the neural network.
     */

    @FXML
    private void onTrainButtonClick() {
        String inputText = labelInput.getText();

        if (inputText == null || inputText.isEmpty())
        {
            resultLabel.setText("Please enter a label.");
            return;
        }



        try {
            int label = Integer.parseInt(inputText);

            if (label < 0 || label > 9) {

                resultLabel.setText("Label must be from 0–9.");
                return;
            }

            double[] input = CanvasUtils.downsample(drawCanvas);
            List<Double> inputVals = new ArrayList<>();
            for (double val : input) inputVals.add(val / 255.0);

            double[] targetArray = new double[10];
            targetArray[label] = 1.0;

            List<Double> targetVals = new ArrayList<>();
            for (double val : targetArray) targetVals.add(val);

            network.feedForward(inputVals);
            network.backProp(targetVals);

            resultLabel.setText("Trained with label: " + label);

        } catch (NumberFormatException e) {
            resultLabel.setText("Invalid number.");
        }
    }

    /**
     * Called to draw on the canvas using the mouse.
     * Draws a circle where the cursor moves for the simulate pen drawing.
     * @param e Mouse drag event.
     */

    private void handleDraw(MouseEvent e) {
        double size = 10.0;
        gc.fillOval(e.getX() - size / 2, e.getY() - size / 2, size, size);
    }

    /**
     * Clears the canvas for the user to draw a new digit.
     */

    private void clearCanvas() {
        gc.clearRect(0, 0, drawCanvas.getWidth(), drawCanvas.getHeight());
    }


    /**
     * Converts the drawing to a 28×28 grayscale
     * The prediction result is shown in the UI.
     */

    private void predictDigit() {
        double[] input = CanvasUtils.downsample(drawCanvas);
        int prediction = network.predict(input);

        resultLabel.setText("Predicted: " + prediction);
    }
}
