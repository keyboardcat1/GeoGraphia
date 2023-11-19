package com.github.keyboardcat1.geographia;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public final class GlobalHeights {

    public static final int WIDTH;
    public static final int HEIGHT;

    private static final List<List<Double>> HEIGHTS = new ArrayList<>();
    private static final String COMMA_DELIMITER = ",";

    static {
        ClassLoader classloader = Thread.currentThread().getContextClassLoader();
        InputStream inputStream = classloader.getResourceAsStream("heights.csv");
        assert inputStream != null;
        InputStreamReader streamReader = new InputStreamReader(inputStream, StandardCharsets.US_ASCII);
        try (BufferedReader reader = new BufferedReader(streamReader)) {

        String line;
        while ((line = reader.readLine()) != null) {
            double[] values = Arrays.stream(line.split(COMMA_DELIMITER)).mapToDouble(Double::parseDouble).toArray();
            HEIGHTS.add(Arrays.stream(values).boxed().toList());
        }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        WIDTH = HEIGHTS.get(0).size();
        HEIGHT = HEIGHTS.size();
    }

    public static double getHeight(int x, int z) {
        return HEIGHTS.get(x).get(z);
    }

}
