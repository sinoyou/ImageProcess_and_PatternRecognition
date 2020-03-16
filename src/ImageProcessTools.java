import transfomer.ExpTransformer;
import transfomer.Transformer;

import java.awt.*;
import java.util.HashMap;

public class ImageProcessTools {

    /**
     * Get Gray Image from color Image with YUV metrics.
     *
     * @param data: pixels of colors.
     * @return grey: gray pixels of original images.
     */
    public static int[][] getGray(Color[][] data) {
        int width = data[0].length;
        int height = data.length;
        int[][] gray = new int[height][width];
        for (int h = 0; h < data.length; h++) {
            for (int w = 0; w < data[0].length; w++) {
                double Y = 0.3 * data[h][w].getRed() +
                        0.59 * data[h][w].getGreen() +
                        0.11 * data[h][w].getBlue();
                gray[h][w] = (int) Y;
            }
        }
        return gray;
    }

    /**
     * Applying Gray Balance algorithm on the gray image.
     *
     * @param data:  src gray image
     * @param scale: How many levels are defined to divide gray value.
     * @return balanced gray image int[][]
     */
    public static int[][] grayBalance(int[][] data, int scale) {
        int[] levelCount = new int[scale];
        double[] originalRate = new double[scale];
        double[] balanceRate = new double[scale];
        int total = data.length * data[0].length;
        // initial
        for (int i = 0; i < scale; i++) {
            levelCount[i] = 0;
        }
        // Count number of different levels
        for (int h = 0; h < data.length; h++) {
            for (int w = 0; w < data[0].length; w++) {
                levelCount[(int) (data[h][w] / (256.0 / scale))] += 1;
            }
        }
        // calculating rate and generating new rate
        for (int i = 0; i < scale; i++) {
            originalRate[i] = levelCount[i] * 1.0 / total;
            balanceRate[i] = i == 0 ? originalRate[i] : balanceRate[i - 1] + originalRate[i];
        }
        // Generating gray balanced image
        int[][] balanced = new int[data.length][data[0].length];
        for (int h = 0; h < data.length; h++) {
            for (int w = 0; w < data[0].length; w++) {
                int level = (int) (data[h][w] / (256.0 / scale));
                balanced[h][w] = (int)(balanceRate[level] * 256);
            }
        }

        clip(balanced);

        return balanced;
    }

    /**
     * Applying Linear Transformation on grey data.
     *
     * @param data        Gray image data.
     * @param transformer Linear Transformer.
     * @return transformed.
     */
    public static int[][] grayTransformation(int[][] data, Transformer transformer, HashMap<String, Double> args) {
        int[][] transformed = transformer.transform(data, args);
        clip(transformed);
        return transformed;
    }

    /**
     * Clip Gray values out of range. (0~255)
     *
     * @param data: values to be clipped.
     */
    private static void clip(int[][] data) {
        int width = data[0].length;
        int height = data.length;

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                if (data[h][w] > 255)
                    data[h][w] = 255;
                else if (data[h][w] < 0)
                    data[h][w] = 0;
            }
        }
    }
}
