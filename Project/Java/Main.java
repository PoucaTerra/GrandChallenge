import java.util.List;

public class Main {

  static final String INPUTFOLDER = "../../Input/";
  static final String OUTPUTFOLDER = "../../Output/";

  static final String RULES = "rule_2M.csv";
  static final String TINYRULES = "rule_tiny.csv";

  static List<Classification> classifications;

  public static void main(String[] args) {
    long time = System.currentTimeMillis();

    Solver s = new Solver(INPUTFOLDER + TINYRULES);
    s.solve(
      INPUTFOLDER + "transactions_tiny.csv",
      OUTPUTFOLDER + "tiny_output.csv"
    );

    // Solver s = new Solver(INPUTFOLDER+RULES);
    // s.solve(INPUTFOLDER+"transactions_0.csv", OUTPUTFOLDER+"output_0.csv");

    // System.out.println(
    //   "Time: " + (System.currentTimeMillis() - time) / 1000.0 + "s"
    // );
  }
}
