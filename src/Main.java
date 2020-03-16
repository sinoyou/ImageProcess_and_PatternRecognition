import io.BmpDumper;
import io.BmpReader;
import transfomer.ExpTransformer;
import transfomer.LinearTransformer;
import transfomer.LogTransformer;
import transfomer.Transformer;

import java.awt.*;
import java.util.HashMap;

public class Main {
    public static void main(String[] args) throws Exception {
        String srcPath = args[0];
        String tarPath = args[1];
        String type = args[2];

        // Load source color/gray image and transformed it into gray image.
        Color[][] colorMap = BmpReader.readImage(srcPath);
        int[][] gray = ImageProcessTools.getGray(colorMap);

        int[][] target;
        // Judge process type
        if(type.equals("gray")){
            System.out.println("[灰度图]");
            target = gray;
        }
        else if (type.equals("balance")) {
            int scale = Integer.parseInt(args[3]);
            System.out.println(String.format("[直方图修正] 离散分级数 %d", scale));
            target = ImageProcessTools.grayBalance(gray, scale);
        }
        else {
            HashMap<String, Double> targs = new HashMap<>();
            Transformer transformer;
            targs.put("a", Double.parseDouble(args[3]));
            targs.put("b", Double.parseDouble(args[4]));
            targs.put("c", Double.parseDouble(args[5]));
            if (args.length > 6) {
                targs.put("d", Double.parseDouble(args[6]));
            }

            switch (type) {
                case "linear":
                    transformer = new LinearTransformer();
                    System.out.println(String.format("[线性变换] a:%.2f b:%.2f, c:%.2f, d:%.2f",
                            targs.get("a"), targs.get("b"), targs.get("c"), targs.get("d")));
                    break;
                case "exp":
                    transformer = new ExpTransformer();
                    System.out.println(String.format("[幂指数变换] a:%.2f b:%.2f, c:%.2f",
                            targs.get("a"), targs.get("b"), targs.get("c")));
                    break;
                case "log":
                    transformer = new LogTransformer();
                    System.out.println(String.format("[对数变换] a:%.2f b:%.2f, c:%.2f",
                            targs.get("a"), targs.get("b"), targs.get("c")));
                    break;
                default:
                    throw new Exception(String.format("No Support for %s", type));
            }

            target = ImageProcessTools.grayTransformation(gray, transformer, targs);
        }

        // Export image
        BmpDumper.dumpGray(target, tarPath);

//        String image = "pandas.bmp";
//        Color[][] colorBmp = BmpReader.readImage(image);
//        int[][] gray = ImageProcessTools.getGray(colorBmp);
//
//        String balance = "balance-pandas.bmp";
//        int[][] balanceGray = ImageProcessTools.grayBalance(gray, 5);
//        BmpDumper.dumpGray(balanceGray, balance);
//
//        String linear = "linear-pandas.bmp";
//        HashMap<String, Double> linerArgs = new HashMap<>();
//        linerArgs.put("a", 0.0);
//        linerArgs.put("b", 224.0);
//        linerArgs.put("c", 32.0);
//        linerArgs.put("d", 255.0);
//        Transformer linearTransformer = new LinearTransformer();
//        int[][] linearGray = linearTransformer.transform(gray, linerArgs);
//        BmpDumper.dumpGray(linearGray, linear);
//
//        String log = "log-pandas.bmp";
//        HashMap<String, Double> logArgs = new HashMap<>();
//        logArgs.put("a", 32.0);
//        logArgs.put("b", 1.0/192);
//        logArgs.put("c", 256.0);
//        Transformer logTransformer = new LogTransformer();
//        int[][] logGray = logTransformer.transform(gray, logArgs);
//        BmpDumper.dumpGray(logGray, log);
//
//        String exp = "exp-pandas.bmp";
//        HashMap<String, Double> expArgs = new HashMap<>();
//        expArgs.put("a", 0.0);
//        expArgs.put("b", 230.0);
//        expArgs.put("c", 1 / 255.0);
//        Transformer expTransformer = new ExpTransformer();
//        int[][] expGray = expTransformer.transform(gray, expArgs);
//        BmpDumper.dumpGray(expGray, exp);
    }

}
