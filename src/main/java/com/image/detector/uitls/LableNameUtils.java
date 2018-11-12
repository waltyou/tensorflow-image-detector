package com.image.detector.uitls;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author waltyou
 * @date 2018/11/06
 */
public class LableNameUtils {

    public static Map<Integer, String> getCategoryIndex(String lableMapPath) throws IOException {
        List<String> lines = Files.readAllLines(Paths.get(lableMapPath));
        Map<Integer, String> id2Label = new HashMap<>(lines.size() / 5);
        Integer id;
        String dispalyName;
        for (int i = 2; i < lines.size(); i = i + 5) {
            id = Integer.valueOf(lines.get(i).split(":")[1].trim());
            dispalyName = lines.get(i + 1).split(":")[1].trim().replaceAll("\"|\"", "");
            id2Label.put(id, dispalyName);
        }
        return id2Label;
    }
}
