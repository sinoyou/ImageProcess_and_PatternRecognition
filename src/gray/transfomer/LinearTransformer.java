package gray.transfomer;

import java.util.HashMap;

public class LinearTransformer implements Transformer {

    @Override
    /*
      线性变换：支持通过(a,b)指定线性变换的区域。
      Keys in args:
      a: lower bound of src zone.
      b: upper bound of src zone.
      c: lower bound of target zone.
      d: upper bound of target zone.
      g(x,y) = a
     */
    public int[][] transform(int[][] gray, HashMap<String, Double> args) {
        int width = gray[0].length;
        int height = gray.length;
        double a = args.get("a");
        double b = args.get("b");
        double c = args.get("c");
        double d = args.get("d");

        int[][] target = new int[height][width];

        for(int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int srcGray = gray[h][w];
                int targetGray;
                if (srcGray < a) {
                    targetGray = (int) c;
                } else if (srcGray > b) {
                    targetGray = (int) d;
                } else {
                    targetGray = (int) ((d - c) / (b - a) * (srcGray - a) + c);
                }
                target[h][w] = targetGray;
            }
        }
        return target;
    }
}
