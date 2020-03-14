import io.BmpDumper;
import io.BmpReader;

import java.awt.*;

public class Main {
    public static void main(String[] args) throws Exception {
        String original_path = "pandas.bmp";
        String gray_path = "gray-" + original_path;
        Color[][] colorBmp = BmpReader.readImage(original_path);
        int[][] gray = ImageProcessTools.getGray(colorBmp);
        BmpDumper.dumpGray(gray, gray_path);
    }
}
