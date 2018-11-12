package com.image.detector;

import com.image.detector.uitls.GraphBuilder;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import static com.image.detector.uitls.ImageNetUtils.getImageNetLabels;
import static com.image.detector.uitls.TensorflowGraphUtils.getGraph;

/**
 * @author waltyou
 * @date 2018/11/05
 */
public class MobileNetClassification {

    private static final String INPUT_LAYER_NAME = "input";
    private static final String OUTPUT_LAYER_NAME = "MobilenetV2/Predictions/Reshape_1";
    private static final String PD_PATH = "src/main/resources/models/mobilenet_v2_1.4_224_frozen.pb";
    private static final int IMAGE_SIZE = 224;
    private static final float MEAN = 255f;

    private static Graph graph;
    private static Map<Integer, String> id2Name;

    public static void main(String[] args) throws IOException {
        String input = "src/main/resources/images/panda.JPG";
        if (!new File(input).exists()) {
            System.out.println("input not exist !!! " + input);
            System.exit(0);
        }

        graph = getGraph(PD_PATH);
        id2Name = getImageNetLabels();

        byte[] imageBytes = Files.readAllBytes(Paths.get(input));
        Tensor<Float> tensor = normalizeImage(imageBytes, IMAGE_SIZE, Float.class);
        float[] result = predict(tensor);

        int maxIndex = getMaxIndex(result);
        System.out.println("Top1 ------------");
        System.out.println(String.format("Object: %s - confidence: %f", id2Name.get(maxIndex), result[maxIndex]));
        System.out.println();

        List<Integer> top5Indexs = getTop5Index(result);
        System.out.println("Top5 ------------");
        for (int i = 4; i >= 0; i--) {
            Integer index = top5Indexs.get(i);
            System.out.println(String.format("Object: %s - confidence: %f", id2Name.get(index), result[index]));
        }
    }

    /**
     * Pre-process input. It resize the image and normalize its pixels
     *
     * @param imageBytes Input image
     * @return Tensor<Float> with shape [1][224][224][3]
     */
    private static <T> Tensor<T> normalizeImage(final byte[] imageBytes, int imageSize, Class<T> type) {
        try (Graph graph = new Graph()) {
            GraphBuilder graphBuilder = new GraphBuilder(graph);
            final Output<Float> output =
                    graphBuilder.div(
                            // Divide each pixels with the MEAN
                            graphBuilder.resizeBilinear(
                                    // Resize using bilinear interpolation
                                    graphBuilder.expandDims(
                                            // Increase the output tensors dimension
                                            graphBuilder.cast(
                                                    // Cast the output to Float
                                                    graphBuilder.decodeJpeg(
                                                            graphBuilder.constant("input", imageBytes), 3),
                                                    type),
                                            graphBuilder.constant("make_batch", 0)),
                                    graphBuilder.constant("size", new int[] {imageSize, imageSize})),
                            graphBuilder.constant("scale", MEAN));

            try (Session session = new Session(graph)) {
                return session.runner().fetch(output.op().name()).run().get(0).expect(type);
            }
        }
    }

    private static float[] predict(Tensor<?> x) {
        try (Session sess = new Session(graph)) {
            try (Tensor y = sess.runner()
                    .feed(INPUT_LAYER_NAME, x)
                    .fetch(OUTPUT_LAYER_NAME)
                    .run().get(0)) {
                int i = (int) y.shape()[1];
                float[][] result = new float[1][i];
                y.copyTo(result);
                return result[0];
            }
        }
    }

    private static int getMaxIndex(float[] result) {
        int maxIndex = 0;
        float maxValue = 0.0f;
        for (int i = 0; i < result.length; i++) {
            if (result[i] > maxValue) {
                maxValue = result[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private static List<Integer> getTop5Index(float[] result) {
        final Integer indexKey = 0;
        final Integer valueKey = 1;
        PriorityQueue<Map<Integer, Float>> priorityQueue = new PriorityQueue<>(6,
                (map1, map2) -> map1.get(valueKey) > map2.get(valueKey) ? 1 : -1);
        for (int i = 0; i < result.length; i++) {
            Map<Integer, Float> map = new HashMap<>(2);
            map.put(indexKey, (float) i);
            map.put(valueKey, result[i]);
            priorityQueue.add(map);
            if (priorityQueue.size() > 5) {
                priorityQueue.remove();
            }
        }
        List<Integer> indexs = new ArrayList<>(5);
        Map<Integer, Float> map;
        while ((map = priorityQueue.poll()) != null) {
            indexs.add(map.get(indexKey).intValue());
        }
        return indexs;
    }

}
