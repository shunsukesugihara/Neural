package com.example.neuralnetworknumber;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;
// HelloApplication.java


import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

/**
 * Start for launching the app.
 * Loads the JavaFX UI and starts the main window.
 * @author Shunsuke Sugihara
 * @since 5/10/2025
 */

public class HelloApplication extends Application {
    @Override


    public void start(Stage stage) throws Exception {
        FXMLLoader fxmlLoader = new FXMLLoader(HelloApplication.class.getResource("hello-view.fxml"));


        Scene scene = new Scene(fxmlLoader.load());


        stage.setTitle("Digit Recognizer");

        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}