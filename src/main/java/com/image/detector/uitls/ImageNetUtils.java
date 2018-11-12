package com.image.detector.uitls;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * @author waltyou
 * @date 2018/11/05
 */
public class ImageNetUtils {

    private static final String METADATA_PATH = "src/main/resources/lableData/imagenet_metadata.txt";
    private static final String SYNSETS_PATH = "src/main/resources/lableData/imagenet_lsvrc_2015_synsets.txt";

    private static Map<Integer, String> label2Names = new HashMap<>(0);

    public static Map<Integer, String> getImageNetLabels() throws IOException {
        if (label2Names.size() > 0) {
            return label2Names;
        }
        Map<String, String> id2Label = new HashMap<>(21843);
        List<String> metadatas = Files.readAllLines(Paths.get(METADATA_PATH));
        for (String line : metadatas) {
            String[] arr = line.split("\t");
            id2Label.put(arr[0], arr[1]);
        }

        Map<Integer, String> result = new HashMap<>(1001);
        result.put(0, "background");
        int labelIndex = 1;
        List<String> synsets = Files.readAllLines(Paths.get(SYNSETS_PATH));
        for (String line : synsets) {
            result.put(labelIndex, id2Label.get(line));
            labelIndex++;
        }
        label2Names = result;
        return result;
    }
}
