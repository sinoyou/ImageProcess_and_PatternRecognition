import transfomer.Transformer;

import java.awt.*;

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

    public static int[][] grayBalance(int[][] data) {
        return null;
    }

    /**
     * Applying Linear Transformation on grey data.
     *
     * @param data        Gray image data.
     * @param transformer Linear Transformer.
     * @return transformed.
     */
    public static int[][] grayLinearTransformation(int[][] data, Transformer transformer) {
        return null;
    }
}
