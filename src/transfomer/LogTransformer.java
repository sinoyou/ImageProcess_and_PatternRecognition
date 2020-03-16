package transfomer;

import java.util.HashMap;

import static java.lang.StrictMath.E;

public class LogTransformer implements Transformer {
    @Override
    /*
      对数数变换：
      Keys in args:
      a: base
      b: scale of diver
      c: diver
      g(x,y) = a + ln(f(x, y) + 1) / (b * ln(c))
     */
    public int[][] transform(int[][] gray, HashMap<String, Double> args) {
        int width = gray[0].length;
        int height = gray.length;
        double a = args.get("a");
        double b = args.get("b");
        double c = args.get("c");

        int[][] target = new int[height][width];

        for(int h = 0; h < height; h++){
            for(int w = 0; w < width; w++){
                int src = gray[h][w];
                target[h][w] = (int)(a + (Math.log(src + 1) / Math.log(E)) / (b * Math.log(c) / Math.log(E)));
            }
        }
        return target;
    }
}
