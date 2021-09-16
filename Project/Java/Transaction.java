import java.util.Arrays;
import java.util.HashSet;

public class Transaction {

  private HashSet<Integer> classification;
  private int[] tuple;

  public Transaction(String[] line) {
    this.classification = new HashSet<>();

    this.tuple = new int[10];
    for (int i = 0; i < 10; i++) {
      this.tuple[i] = Integer.valueOf(line[i]);
    }
  }

  public void addClassification(int classification) {
    this.classification.add(classification);
  }

  public int[] getTuple() {
    return tuple;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < tuple.length; i++) {
      sb.append(tuple[i] + ";");
    }
    return sb.toString();
  }
}
