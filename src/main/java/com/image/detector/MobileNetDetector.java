package com.image.detector;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.framework.ConfigProto;
import org.tensorflow.framework.GPUOptions;
import org.tensorflow.types.UInt8;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static com.image.detector.uitls.LableNameUtils.getCategoryIndex;
import static com.image.detector.uitls.TensorflowGraphUtils.getGraph;

/**
 * @author waltyou
 * @date 2018/11/06
 */
public class MobileNetDetector {

    private static final String INPUT_LAYER_NAME = "image_tensor";
    private static final String PD_PATH = "src/main/resources/models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb";
    private static final String LABEL_PATH = "src/main/resources/lableData/mscoco_label_map.pbtxt";

    private static final String[] LAYER_NAMES = {"num_detections", "detection_boxes",
            "detection_scores", "detection_classes"};

    private static Graph graph;
    private static Map<Integer, String> id2Name;
    private static byte[] config;

    public static void main(String[] args) throws IOException {
        String input = "src/main/resources/images/beach.jpg";
        String output = "/tmp/tmp.jpg";

        graph = getGraph(PD_PATH);
        id2Name = getCategoryIndex(LABEL_PATH);
        config = ConfigProto.newBuilder()
                .setGpuOptions(GPUOptions.newBuilder()
                        .setAllowGrowth(true)
                        .setPerProcessGpuMemoryFraction(0.04)
                        .build()
                )
                .setAllowSoftPlacement(true)
                .build().toByteArray();

        Tensor<UInt8> tensor = makeImageTensor(input);
        System.out.println(Arrays.toString(tensor.shape()));
        List<Tensor<?>> results = predict(tensor);

        int numDetections = getNumDetctions(results.get(0));
        int maxObjects = (int) results.get(1).shape()[1];
        float[][] boxes = results.get(1).copyTo(new float[1][maxObjects][4])[0];
        float[] scores = results.get(2).copyTo(new float[1][maxObjects])[0];
        float[] classes = results.get(3).copyTo(new float[1][maxObjects])[0];
        for (int i = 0; i < numDetections; i++) {
            int id = (int) classes[i];
            String name = id2Name.get(id);
            float score = scores[i];
            System.out.println(name + " : " + score);
        }
        drawBox(numDetections, boxes, scores, classes, input, output);
    }

    private static Tensor<UInt8> makeImageTensor(String filename) throws IOException {
        BufferedImage img = ImageIO.read(new File(filename));
        if (img.getType() != BufferedImage.TYPE_3BYTE_BGR) {
            throw new IOException(
                    String.format(
                            "Expected 3-byte BGR encoding in BufferedImage, found %d (file: %s). This code could be made more robust",
                            img.getType(), filename));
        }
        byte[] data = ((DataBufferByte) img.getData().getDataBuffer()).getData();
        // ImageIO.read seems to produce BGR-encoded images, but the model expects RGB.
        bgr2rgb(data);
        final long BATCH_SIZE = 1;
        final long CHANNELS = 3;
        long[] shape = new long[]{BATCH_SIZE, img.getHeight(), img.getWidth(), CHANNELS};
        return Tensor.create(UInt8.class, shape, ByteBuffer.wrap(data));
    }

    private static void bgr2rgb(byte[] data) {
        for (int i = 0; i < data.length; i += 3) {
            byte tmp = data[i];
            data[i] = data[i + 2];
            data[i + 2] = tmp;
        }
    }

    private static List<Tensor<?>> predict(Tensor<UInt8> x) {
        try (Session sess = new Session(graph, config)) {
            Session.Runner runner = sess.runner().feed(INPUT_LAYER_NAME, x);
            for (String layerName : LAYER_NAMES) {
                runner.fetch(layerName);
            }
            return runner.run();
        }
    }

    private static int getNumDetctions(Tensor<?> tensor) {
        float[] x = new float[1];
        tensor.copyTo(x);
        return (int) x[0];
    }

    private static void drawBox(int numDetections, float[][] boxes, float[] detectionScores,
                                float[] detectionClasses, String input, String output) throws IOException {
        File image = new File(input);
        BufferedImage bufferedImage = ImageIO.read(image);
        int w = bufferedImage.getWidth();
        int h = bufferedImage.getHeight();
        Graphics2D graphics = (Graphics2D) bufferedImage.getGraphics();
        for (int i = 0; i < numDetections; i++) {
            float[] box = boxes[i];
            float ymin = box[0];
            float xmin = box[1];
            float ymax = box[2];
            float xmax = box[3];
            int x = (int) (xmin * w);
            int width = (int) ((xmax - xmin) * w);
            int y = (int) (ymin * h);
            int heigth = (int) ((ymax - ymin) * h);

            int id = (int) detectionClasses[i];
            String name = id2Name.get(id);
            float score = detectionScores[i];
            graphics.setColor(Color.WHITE);
            graphics.drawString(name + " " + score, x, y - 7);
            graphics.setColor(Color.GREEN);
            graphics.drawRect(x, y, width, heigth);
        }
        graphics.dispose();
        ImageIO.write(bufferedImage, "jpg", new File(output));
    }

}
