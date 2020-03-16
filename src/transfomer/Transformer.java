package transfomer;

import java.util.HashMap;

public interface Transformer {
   int[][] transform(int[][] gray, HashMap<String, Double> args);
}
