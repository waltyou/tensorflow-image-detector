package com.image.detector.uitls;

import org.tensorflow.Graph;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * @author waltyou
 * @date 2018/11/06
 */
public class TensorflowGraphUtils {

    public static Graph getGraph(String pdPath) throws IOException {
        Graph graph = new Graph();
        graph.importGraphDef(Files.readAllBytes(Paths.get(pdPath)));
        return graph;
    }


}
