package gray.io;

import java.awt.*;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class BmpReader {

    public static Color[][] readImage(String path) throws Exception {
        Color[][] data;
        byte[] temp = new byte[4];

        // Create File Input Stream
        FileInputStream fileInputStream = new FileInputStream(path);
        BufferedInputStream bis = new BufferedInputStream(fileInputStream);

        bis.mark(0);

        // Read Data Offset from file head
        // bis.read(temp, 0xa, 4);
        read4(bis, 0xa, temp);
        int dataOffset = byte2Int(temp);

        // Read Image Width and Height
        // bis.read(temp, 0x12, 4);
        read4(bis, 0x12, temp);
        int width = byte2Int(temp);
        // bis.read(temp, 0x16, 4);
        read4(bis, 0x16, temp);
        int height = byte2Int(temp);
        data = new Color[height][width];
        int skip = 4 - width * 3 % 4;

        // Check Quantization Support
        // bis.read(temp, 0x1c, 4);
        read4(bis, 0x1c, temp);
        if (!(temp[0] == 0x18 && temp[1] == 0 && temp[2] == 0 && temp[3] == 0)) {
            throw new Exception("gray.io.BmpReader only support 24 bit quantization without compressed.");
        }

        // Start Read Image Data
        bis.reset();
        bis.skip(dataOffset);
        /*
        (0,0) ------ width
          |             |
          |             |
          |             |
        height ----- (height-1, width-1)
        */
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int blue = bis.read();
                int green = bis.read();
                int red = bis.read();
                Color rgb = new Color(red, green, blue);
                data[h][w] = rgb;
            }
            if (skip != 4) {
                bis.skip(skip);
            }
        }

        bis.close();
        fileInputStream.close();
        return data;
    }

    private static void read4(BufferedInputStream bis, int offset, byte[] b) throws IOException {
        bis.reset();
        bis.skip(offset);
        bis.read(b);
    }

    private static int byte2Int(byte[] bytes) {
        int i = ((bytes[3] & 0xff) << 24) | ((bytes[2] & 0xff) << 16) | ((bytes[1] & 0xff) << 8) | ((bytes[0] & 0xff));
        return i;
    }
}
