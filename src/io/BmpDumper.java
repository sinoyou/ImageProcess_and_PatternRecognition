package io;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class BmpDumper{
    public static void dumpGray(int[][] data, String path) throws IOException {
        BufferedImage gray =
                new BufferedImage(data[0].length, data.length, BufferedImage.TYPE_INT_RGB);

        for(int h = 0; h < data.length; h++){
            for(int w = 0; w < data[0].length; w++){
                Color g = new Color(data[h][w], data[h][w], data[h][w]);
                gray.setRGB(w, data.length - h - 1, g.getRGB());
            }
        }

        File output = new File(path);
        ImageIO.write(gray, "bmp", output);
    }
}
