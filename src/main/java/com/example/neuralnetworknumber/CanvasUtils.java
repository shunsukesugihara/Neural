package com.example.neuralnetworknumber;

import javafx.scene.canvas.Canvas;
import javafx.scene.image.PixelReader;
import javafx.scene.image.WritableImage;

/**
 * This the processes canvas images to make them usable
 * by the neural network for training
 * @author Shunsuke Sugihara
 * @since 5/10/2025
 */

public class CanvasUtils {


    /**
     * Converts the image drawn on the canvas into a 28×28 grayscale pixel
     * Each pixel is represented by a value from 0 to 255 based on brightness.
     * This becomes the input for the neural network.
     *
     * @param canvas
     * @return 784-pixel array of grayscale value
     */


    public static double[] downsample(Canvas canvas) {
        WritableImage snapshot = canvas.snapshot(null, null);
        int size = 28;
        double[] pixels = new double[size * size];

        snapshot = new WritableImage(snapshot.getPixelReader(), (int) canvas.getWidth(), (int) canvas.getHeight());

        snapshot = new WritableImage(snapshot.getPixelReader(), size, size);
        PixelReader reader = snapshot.getPixelReader();

        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                int argb = reader.getArgb(x, y);
                int gray = (argb >> 16) & 0xFF;
                pixels[y * size + x] = gray;
            }
        }

        return pixels;
    }
}