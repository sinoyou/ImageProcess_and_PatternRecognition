package transfomer;

import java.util.HashMap;

public class ExpTransformer implements Transformer {
    @Override
    /*
      幂指数变换
      Keys in args:
      a: base
      b: power base
      c: scale
      g(x,y) = b ** (c * (f(x,y) - a)) - 1
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
                target[h][w] = (int)(Math.pow(b, c * (src - a)) - 1);
            }
        }
        return target;
    }
}
